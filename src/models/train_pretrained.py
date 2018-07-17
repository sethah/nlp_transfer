%cd nlp_transfer/src
%load_ext autoreload
%autoreload 2
%matplotlib inline

sys.path.append("/home/cdsw/pytorch-openai-transformer-lm/")
import numpy as np
import math, copy, time
import matplotlib.pyplot as plt
import seaborn
import os
import itertools
import spacy
import json
from collections import Counter
import ftfy
import re

os.chdir(os.getcwd())

# pytorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data, datasets
from torchtext.vocab import Vocab
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers.checkpoint import ModelCheckpoint

# local packages
from models.models import *
from models.load_pretrain import load_model
import models.utils
from models.text_utils import TextEncoder, AttentionIterator
from models.metrics import Perplexity


SAVE_PATH = "/tmp/ignite2"
CHECKPOINT_DIR = "/tmp/ignite"
MODELS_PATH = "../models/openai-transformer-lm/"
ENCODER_PATH = os.path.join(MODELS_PATH, "encoder_bpe_40000.json")
BPE_PATH = "/home/cdsw/pytorch-openai-transformer-lm/model/vocab_40000.bpe"

BATCH_SIZE = 512
BPTT_LEN = 70
PAD_TOKEN = "<pad>"

num_gpus = torch.cuda.device_count()
devices = [torch.device("cuda", i) for i in range(num_gpus)]
device_ids = [d.index for d in devices]

encoder_dict = json.load(open(ENCODER_PATH))
encoder_dict = {k: len(encoder_dict) - v for k, v in encoder_dict.items()}
vocab = Vocab(Counter(encoder_dict), specials=[])
special_embeds = 3
pos_start = len(vocab) + special_embeds

nlp = spacy.load('en', disable=['parser', 'tagger', 'ner', 'textcat'])
text_encoder = TextEncoder(ENCODER_PATH, BPE_PATH)
def tokenizer(text):
  text = text.replace("<unk>", "unk")
  tokens = []
  for tok in nlp(models.utils.text_standardize(ftfy.fix_text(text))):
    tokens.extend(text_encoder.bpe(tok.text).split(' '))
  return tokens

TEXT = data.Field(lower=True, tokenize=tokenizer)
train, valid, test = datasets.WikiText2.splits(TEXT)
TEXT.vocab = vocab
pad_idx = vocab.stoi[PAD_TOKEN]
iters = data.BPTTIterator.splits(
    (train, valid, test),
    batch_size=BATCH_SIZE,
    bptt_len=BPTT_LEN,
    device=0,
    repeat=False)
train_iter, valid_iter, test_iter = (AttentionIterator(it, pos_start, pad_idx) for it in iters)

#def rebatch(pad_idx, batch, pos_start):
#    "Fix order in torchtext to match ours"
#    src = batch.text.transpose(0, 1)
#    batch_size, seq_len = src.shape
#    b = utils.Batch(batch.text.transpose(0, 1), batch.target.transpose(0, 1), pad_idx)
#    position_indices = torch.arange(pos_start, pos_start + seq_len).repeat(batch_size, 1).long().to(b.src.device)
#    b.src = torch.stack((b.src, position_indices), dim=2)
#    return b


embeds, encoder, generator = load_model(special_embeds=special_embeds, weights_path=MODELS_PATH)
criterion = nn.NLLLoss().to(devices[0])
valid_criterion = nn.NLLLoss(reduce=False, size_average=False).to(devices[0])

# freeze all parameters but the generator
for param in encoder.parameters():
  param.requires_grad = False
for param in embeds.parameters():
  param.requires_grad = False

class EncoderOnly(nn.Module):
    def __init__(self, encoder, embed, generator):
        super(EncoderOnly, self).__init__()
        self.encoder = encoder
        self.embed = embed
        self.generator = generator
        
    def forward(self, src, src_mask):
        "Take in and process masked src and target sequences."
        embeds = self.embed(src)
        return self.encoder.forward(embeds.sum(dim=2), src_mask)

model = EncoderOnly(encoder, embeds, generator).to(devices[0])
model_par = nn.DataParallel(model, device_ids=[d.index for d in devices])
parameters = filter(lambda p: p.requires_grad, model.parameters())
frozen = filter(lambda p: not p.requires_grad, model.parameters())
model_opt = torch.optim.Adam(parameters, lr=0.01, betas=(0.9, 0.98), eps=1e-9)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(model_opt, gamma=0.95)

def training_update_function(engine, batch):
  model_par.train()
  out = model_par.forward(batch.src, batch.src_mask)
  loss_compute = models.utils.MultiGPULossCompute(model.generator, criterion, device_ids, opt=model_opt, clip=0.125)
  loss = loss_compute(out, batch.src_y, parameters)
  
  return loss

def inference(engine, batch):
  model_par.eval()
  out = model_par.forward(batch.src, batch.src_mask)
  return out, batch.src_y, batch.ntokens.float().item()

trainer = Engine(training_update_function)

metric = Perplexity(models.utils.MultiGPULossCompute(model.generator, valid_criterion, device_ids, opt=None))
evaluator = Engine(inference)
metric.attach(evaluator, "ppl")

@trainer.on(Events.ITERATION_COMPLETED)
def log_training_loss(trainer):
    if (trainer.state.iteration + 1) % 20 == 0:
      print("Epoch[{}] Iteration[{}] Loss: {:.2f}".format(trainer.state.epoch, 
                                                          trainer.state.iteration + 1,
                                                          trainer.state.output))

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(valid_iter)
    loss = evaluator.state.metrics['ppl']
    print("Training Results - Epoch: {}  Valid loss: {:.2f}"
          .format(trainer.state.epoch, loss))
    
@trainer.on(Events.ITERATION_COMPLETED)
def lr_step(trainer):
    if (trainer.state.iteration + 1) % 50 == 0:
      lr_scheduler.step()
      print("Epoch[%d] Iteration[%d] New learning rates: %s" % (trainer.state.epoch,
                                                               trainer.state.iteration + 1,
                                                               lr_scheduler.get_lr()))

#checkpointer = ModelCheckpoint(CHECKPOINT_DIR, "my_model18", save_interval=10,
#                              create_dir=True)
#trainer.add_event_handler(Events.ITERATION_COMPLETED, checkpointer, {'model': model, 'opt': model_opt.optimizer})

train_state = trainer.run(train_iter, max_epochs=5)
#evaluator.run(valid_iter)

torch.load(os.path.join(CHECKPOINT_DIR, "my_model13_opt_20.pth"))
    
torch.save(model.state_dict(), os.path.join(SAVE_PATH))
torch.save(model_opt.state_dict(), os.path.join(SAVE_PATH, "model_opt"))
    

vb = next(iter(valid_iter))
vb = rebatch(pad_idx, vb, pos_start)

src_text = [[vocab.itos[i] for i in idx] for idx in vb.src[:, :, 0]]
trg_text = [[vocab.itos[i] for i in idx] for idx in vb.src_y]
feat = model_par.forward(vb.src, vb.src_mask)
#log_probs = model.generator.forward(feat)

valid_criterion = nn.NLLLoss(reduce=False, size_average=False).to(devices[0])
valid_loss_compute = utils.MultiGPULossCompute(model.generator, valid_criterion, device_ids, opt=None)

lsum = 0.0
nsum = 0
for vb in valid_iter:
  vb = rebatch(pad_idx, vb, pos_start)
  feat = model_par.forward(vb.src, vb.src_mask)
  nll = valid_loss_compute(feat, vb.src_y, vb.ntokens.float())
  lsum += nll
  nsum += vb.ntokens.float().item()
  
print(lsum / nsum)


g = log_probs.contiguous().view(-1, log_probs.size(-1)) 
t = vb.src_y[:10].contiguous().view(-1).to("cuda:0")
perp.eval_batch(g, t)
perp.get_loss()


preds = torch.max(log_probs, dim=2)
pred_text = [[vocab.itos[i] for i in idx] for idx in preds[1]]
i = 1
list(zip(src_text[i], pred_text[i], trg_text[i]))

sample = [b for b in itertools.islice(train, 5)]

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
  

