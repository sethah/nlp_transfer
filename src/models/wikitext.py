import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
import os
import itertools
seaborn.set_context(context="talk")
%matplotlib inline

%load_ext autoreload
%autoreload 2

from models import *

from torchtext import data, datasets
import spacy

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

class GPULossCompute(object):     
    "A multi-gpu loss compute and train function."
    def __init__(self, generator, criterion, devices, opt=None):
        # Send out to different gpus.
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion, 
                                               devices=devices)
        self.opt = opt
        self.devices=devices
        
    def __call__(self, out, targets, normalize):
        total = 0.0
        generator = self.generator

        # Predict distributions
        generator = nn.parallel.replicate(self.generator, devices=self.devices)
        out_scatter = nn.parallel.scatter(out, target_gpus=self.devices)
        targ_scatter = nn.parallel.scatter(targets, target_gpus=self.devices)
        
        gen = nn.parallel.parallel_apply(generator, [(o,) for o in out_scatter])
        
        # Compute loss. 
        y = [(g.contiguous().view(-1, g.size(-1)), 
                  t.contiguous().view(-1)) for g, t in zip(gen, targ_scatter)]

        # Sum and normalize loss
        loss = nn.parallel.parallel_apply(self.criterion, y)
        l = nn.parallel.gather([l.view(-1) for l in loss], target_device=self.devices[0])
        l = l.sum()
        total += l.item()

        # Backprop loss to output of transformer
        if self.opt is not None:
            l.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()

        return total
  
class Batch(object):
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, tgt, pad=0):
        self.src = src
        self.src_y = tgt
        self.src_mask = \
            self.make_std_mask(self.src, pad)
        self.ntokens = (self.src_y != pad).sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask)
        return tgt_mask

c = copy.deepcopy
BATCH_SIZE = 1024

SRC = data.Field(tokenize="spacy", pad_token="<pad>")
TEXT = data.Field(lower=True, tokenize="spacy")
train, valid, test = datasets.WikiText2.splits(TEXT)
TEXT.build_vocab(train, vectors="glove.6B.200d")
train_iter, valid_iter, test_iter = data.BPTTIterator.splits(
    (train, valid, test),
    batch_size=BATCH_SIZE,
    bptt_len=30,
    device=0,
    repeat=False)
vocab = TEXT.vocab
pad_idx = vocab.stoi["<pad>"]

h = 8
d_model = vocab.vectors.shape[1]
d_ff = 1024
dropout = 0.3
n_layers = 6

attn = MultiHeadedAttention(h, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
position = PositionalEncoding(d_model, dropout)
encoder = Encoder(EncoderLayer(d_model, attn, ff, dropout), n_layers)
embeds = Embeddings(d_model, len(vocab))
embeds.lut.weight.requires_grad = False
embeds.lut.weight.copy_(vocab.vectors)
pos_embeds = nn.Sequential(embeds, c(position))
generator = Generator(d_model, len(vocab))
criterion = LabelSmoothing(size=len(vocab), padding_idx=pad_idx, smoothing=0.1)
criterion.to("cuda:0")

class EncoderOnly(nn.Module):
    def __init__(self, encoder, embed, generator):
        super(EncoderOnly, self).__init__()
        self.encoder = encoder
        self.embed = embed
        self.generator = generator
        
    def forward(self, src, src_mask):
        "Take in and process masked src and target sequences."
        return self.encoder.forward(self.embed(src), src_mask)
      
def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src = batch.text.transpose(0, 1)
    return Batch(batch.text.transpose(0, 1), batch.target.transpose(0, 1), pad_idx)
  
num_gpus = torch.cuda.device_count()
devices = list(range(num_gpus))
model = EncoderOnly(encoder, pos_embeds, generator).to("cuda:0")
parameters = filter(lambda p: p.requires_grad, model.parameters())
frozen = filter(lambda p: not p.requires_grad, model.parameters())
model_opt = NoamOpt(model.embed[0].d_model, 2, 1000,
        torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9))
trainer = create_supervised_trainer(model, optimizer, loss)

SAVE_PATH = "/home/cdsw/"
#model.load_state_dict(torch.load(SAVE_PATH))
model_par = nn.DataParallel(model, device_ids=devices)

def run_epoch(data_iter, model, loss_compute):
  start = time.time()
  total_tokens = 0
  total_loss = 0
  tokens = 0
  for i, batch in enumerate(data_iter):
    out = model_par.forward(batch.src, batch.src_mask)
    loss = loss_compute(out, batch.src_y, batch.ntokens.float())
    total_loss += loss
    total_tokens += batch.ntokens.item()
    tokens += batch.ntokens.item()
    if (i + 1) % 10 == 0:
      elapsed = time.time() - start
      print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
              (i + 1, loss / batch.ntokens.float(), tokens / elapsed))
      start = time.time()
      tokens = 0
  return total_loss / total_tokens

for epoch in range(10):
    model_par.train()
    run_epoch((rebatch(pad_idx, b) for b in train_iter),
             model_par, GPULossCompute(model.generator, criterion, devices, opt=model_opt))
    model_par.eval()
    valid_loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter),
             model_par, GPULossCompute(model.generator, 
                                       criterion, devices, opt=None))
    print("Optimizer: %0.5f, %d" % (model_opt.rate(), model_opt._step))
    print("Valid loss: ", valid_loss)
    
torch.save(model.state_dict(), os.path.join(SAVE_PATH, "model"))
torch.save(model_opt.state_dict(), os.path.join(SAVE_PATH, "model_opt"))
    

vb = next(iter(valid_iter))
vb = rebatch(pad_idx, vb)

src_text = [[vocab.itos[i] for i in idx] for idx in vb.src]
trg_text = [[vocab.itos[i] for i in idx] for idx in vb.src_y]
feat = model.forward(vb.src[:10], vb.src_mask[:10])
log_probs = model.generator.forward(feat)

g = log_probs.contiguous().view(-1, log_probs.size(-1)) 
t = vb.src_y[:10].contiguous().view(-1).to("cuda:0")
perp.eval_batch(g, t)
perp.get_loss()


preds = torch.max(log_probs, dim=2)
pred_text = [[vocab.itos[i] for i in idx] for idx in preds[1]]
i = 8
list(zip(src_text[i], pred_text[i], trg_text[i]))

query = """as all the care of her aged mother devolved upon her , but she did manage to b
egin planning a stained glass window design in her""".split(" ")
query = torch.LongTensor([vocab.stoi[w] for w in query]).to("cuda:0").view(1, -1)

feat = model.forward(query, Batch.make_std_mask(query, pad_idx).to("cuda:0"))
log_probs = model.generator.forward(feat)
preds = torch.max(log_probs, dim=2)
pred_text = [[vocab.itos[i] for i in idx] for idx in preds[1]]
pred_text

preds = torch.max(log_probs, dim=2)
pred_text = [[vocab.itos[i] for i in idx] for idx in preds[1]]
pred_text

perp = Perplexity(weight=torch.ones(len(vocab)).to("cuda:0"), mask=pad_idx)
for batch in (rebatch(pad_idx, b) for b in valid_iter):
  out = model.forward(batch.src, batch.src_mask)
  log_probs = model.generator.forward(out)
  g = log_probs.contiguous().view(-1, log_probs.size(-1)) 
  t = batch.src_y.contiguous().view(-1)
  perp.eval_batch(g, t)
  

