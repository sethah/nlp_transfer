%load_ext autoreload
%autoreload 2
import numpy as np
import math, copy, time
import matplotlib.pyplot as plt
import seaborn
import os
import itertools
import spacy

# pytorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data, datasets
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers.checkpoint import ModelCheckpoint

# local packages
from models import *
import utils

seaborn.set_context(context="talk")
%matplotlib inline

BATCH_SIZE = 1024
BPTT_LEN = 30
PAD_TOKEN = "<pad>"
CHECKPOINT_DIR = "/tmp/ignite"

num_gpus = torch.cuda.device_count()
devices = [torch.device("cuda", i) for i in range(num_gpus)]
device_ids = [d.index for d in devices]

TEXT = data.Field(lower=True, tokenize="spacy")
train, valid, test = datasets.WikiText2.splits(TEXT)
TEXT.build_vocab(train, vectors="glove.6B.200d")
train_iter, valid_iter, test_iter = data.BPTTIterator.splits(
    (train, valid, test),
    batch_size=BATCH_SIZE,
    bptt_len=BPTT_LEN,
    device=0,
    repeat=False)
vocab = TEXT.vocab
pad_idx = vocab.stoi[PAD_TOKEN]

# model parameters
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
pos_embeds = nn.Sequential(embeds, position)
generator = Generator(d_model, len(vocab))
criterion = LabelSmoothing(size=len(vocab), padding_idx=pad_idx, smoothing=0.1).to(devices[0])

class EncoderOnly(nn.Module):
    def __init__(self, encoder, embed, generator):
        super(EncoderOnly, self).__init__()
        self.encoder = encoder
        self.embed = embed
        self.generator = generator
        
    def forward(self, src, src_mask):
        "Take in and process masked src and target sequences."
        return self.encoder.forward(self.embed(src), src_mask)

model = EncoderOnly(encoder, pos_embeds, generator).to(devices[0])
model_par = nn.DataParallel(model, device_ids=[d.index for d in devices])
parameters = filter(lambda p: p.requires_grad, model.parameters())
frozen = filter(lambda p: not p.requires_grad, model.parameters())
sub_optimizer = torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9)
model_opt = NoamOpt(model.embed[0].d_model, 2, 1000, sub_optimizer)

def training_update_function(engine, batch):
  model_par.train()
  b = utils.rebatch(pad_idx, batch)
  out = model_par.forward(b.src, b.src_mask)
  loss_compute = utils.MultiGPULossCompute(model.generator, criterion, device_ids, opt=model_opt)
  loss = loss_compute(out, b.src_y, b.ntokens.float())
  
  return loss

def inference(engine, batch):
  model_par.eval()
  b = utils.rebatch(pad_idx, batch)
  out = model_par.forward(b.src, b.src_mask)
  loss_compute = utils.MultiGPULossCompute(model.generator, criterion, device_ids, opt=None)
  loss = loss_compute(out, b.src_y, b.ntokens.float())
  
  return loss

trainer = Engine(training_update_function)
evaluator = Engine(inference)

@trainer.on(Events.ITERATION_COMPLETED)
def log_training_loss(trainer):
    print("Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output))

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    valid_loader = valid_iter
    evaluator.run(valid_loader)
    loss = evaluator.state.output
    print("Training Results - Epoch: {}  Valid loss: {:.2f}"
          .format(trainer.state.epoch, loss))
#
#checkpointer = ModelCheckpoint(CHECKPOINT_DIR, "my_model13", save_interval=10,
#                              create_dir=True)
#trainer.add_event_handler(Events.ITERATION_COMPLETED, checkpointer, {'model': model, 'opt': model_opt.optimizer})

train_loader = train_iter
train_state = trainer.run(train_loader, max_epochs=10)

torch.load(os.path.join(CHECKPOINT_DIR, "my_model13_opt_20.pth"))
    
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
  

