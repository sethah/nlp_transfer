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
from collections import Counter, defaultdict
import ftfy
import re

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
from models.text_utils import TextEncoder, AttentionIterator, Batch
from models.metrics import Perplexity
from models.opt import OpenAIAdam


SAVE_PATH = "/tmp/ignite2"
CHECKPOINT_DIR = "/tmp/ignite"
MODELS_PATH = "../models/openai-transformer-lm/"
ENCODER_PATH = os.path.join(MODELS_PATH, "encoder_bpe_40000.json")
BPE_PATH = "/home/cdsw/pytorch-openai-transformer-lm/model/vocab_40000.bpe"

BATCH_SIZE = 128
BPTT_LEN = 70
PAD_TOKEN = "<pad>"

num_gpus = torch.cuda.device_count()
devices = [torch.device("cuda", i) for i in range(num_gpus)]
device_ids = [d.index for d in devices]

encoder_dict = json.load(open(ENCODER_PATH))
special_embeds = 2
pos_start = len(encoder_dict) + special_embeds
encoder_dict['<eos>'] = len(encoder_dict)
encoder_dict[PAD_TOKEN] = len(encoder_dict)
encoder_dict = {k: len(encoder_dict) - v for k, v in encoder_dict.items()}
cnt = Counter(encoder_dict)
vocab = Vocab(cnt, specials=[])

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

embeds, encoder, generator = load_model(special_embeds=special_embeds, weights_path=MODELS_PATH)
criterion = nn.NLLLoss(reduce=False, size_average=False).to(devices[0])
valid_criterion = nn.NLLLoss(reduce=False, size_average=False).to(devices[0])

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
#model_opt = torch.optim.Adam(parameters, lr=0.001, betas=(0.9, 0.98), eps=1e-9)
#lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(model_opt, gamma=0.95)
#model_opt = torch.optim.SGD(parameters, lr=1)
#clr = models.utils.cyclical_lr(100, min_lr=0.05 / 10, max_lr=0.05, mode='triangular')
#lr_scheduler = torch.optim.lr_scheduler.LambdaLR(model_opt, [clr])

def training_update_function(engine, batch):
  model_par.train()
  out = model_par.forward(batch.src, batch.src_mask)
  loss_compute = models.utils.MultiGPULossCompute(model.generator, criterion, device_ids)
  loss = loss_compute(out, batch.src_y)
  loss.backward()
  model_opt.step()
  model_opt.zero_grad()
  
  return loss.item() / np.prod(batch.src_y.shape)

def inference(engine, batch):
  model_par.eval()
  out = model_par.forward(batch.src, batch.src_mask)
  return out, batch.src_y, batch.ntokens.float().item()

trainer = Engine(training_update_function)

metric = Perplexity(models.utils.MultiGPULossCompute(model.generator, valid_criterion, device_ids))
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
    
@trainer.on(Events.EPOCH_COMPLETED)
def unfreeze_layers(trainer):
  unfreeze(param_groups[:trainer.state.epoch + 1])
  trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
  opt_params = [{'params': p, 'lr': (0.01 if i == 0 else 0.01 / (i * 2)) * 0.95**(trainer.state.epoch - i)} for i, p in enumerate(param_groups[:trainer.state.epoch + 1])]
  print(f'Now training {len(trainable_params)} parameters')
  model_opt.add_param_group(opt_params[-1])
  for pg, lr in zip(model_opt.param_groups, [x['lr'] for x in opt_params]):
    pg['lr'] = lr
#  clr = models.utils.cyclical_lr(150, min_lr=0.1, max_lr=1, mode='triangular')
#  lr_scheduler = torch.optim.lr_scheduler.LambdaLR(model_opt, [clr] * len(opt_params))
  lr_scheduler.lr_lambdas.append(models.utils.cyclical_lr(150, min_lr=0.1, max_lr=1, mode='triangular'))
  lr_scheduler.last_epoch = -1
  lr_scheduler.base_lrs = list(map(lambda group: group['lr'], model_opt.param_groups))
  for param_group in model_opt.param_groups:
    print(param_group['lr'])
#
@trainer.on(Events.ITERATION_COMPLETED)
def lr_step(trainer):
    if (trainer.state.iteration + 1) % 1 == 0:
      lr_scheduler.step()
      print("Epoch[%d] Iteration[%d] New learning rates: %s" % (trainer.state.epoch,
                                                               trainer.state.iteration + 1,
                                                               lr_scheduler.get_lr()))

param_groups = [list(generator.parameters()),
  [plist for l in encoder.layers[::-1][0:3] for plist in list(l.parameters())],
  [plist for l in encoder.layers[::-1][3:6] for plist in list(l.parameters())],
  [plist for l in encoder.layers[::-1][6:9] for plist in list(l.parameters())],
  [plist for l in encoder.layers[::-1][9:12] for plist in list(l.parameters())],
  list(embeds.parameters())]
opt_params = [{'params': p, 'lr': 0.01 if i == 0 else 0.01 / (i * 2)} for i, p in enumerate(param_groups[:1])]
model_opt = OpenAIAdam(opt_params[:1],
                     lr=0.01,
                     schedule='warmup_linear',
                     warmup=0.005,
                     t_total=len(train_iter.iterator) * 1,
                     b1=0.8,
                     b2=0.99,
                     e=1e-8,
                     l2=0.01,
                     vector_l2='store_true',
                     max_grad_norm=0.25)

clr = models.utils.cyclical_lr(150, min_lr=0.1, max_lr=1, mode='triangular')
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(model_opt, [clr])

#checkpointer = ModelCheckpoint(CHECKPOINT_DIR, "my_model", save_interval=1,
#                              create_dir=True, require_empty=False)
#trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model, 'opt': model_opt})

def unfreeze(param_groups):
  for param_group in param_groups:
    for p in param_group:
      p.requires_grad = True
      
def freeze(param_groups):
  for param_group in param_groups:
    for p in param_group:
      p.requires_grad = False
      
freeze(param_groups[1:])
num_epochs = 5
train_state = trainer.run(train_iter, max_epochs=num_epochs)


evaluator.run(test_iter)

torch.load(os.path.join(CHECKPOINT_DIR, "my_model13_opt_20.pth"))
    
torch.save(model.state_dict(), os.path.join("/tmp/mymodel"))
torch.save(model_opt.state_dict(), os.path.join("/tmp/model_opt"))
    

vb = next(iter(valid_iter))

src_text = [[vocab.itos[i] for i in idx] for idx in vb.src[:, :, 0]]
trg_text = [[vocab.itos[i] for i in idx] for idx in vb.src_y]
feat = model_par.forward(vb.src, vb.src_mask)
#log_probs = model.generator.forward(feat)

valid_criterion = nn.NLLLoss(reduce=False, size_average=False).to(devices[0])
valid_loss_compute = models.utils.MultiGPULossCompute(model.generator, valid_criterion, device_ids)

feat = model.forward(vb.src[:16], vb.src_mask[:16])
out = model.generator.forward(feat)
preds = torch.argmax(out, dim=2)
preds.shape
y = vb.src_y[:16]

src_text = [[vocab.itos[i] for i in idx] for idx in vb.src[:16, :, 0]]
pred_text = [[vocab.itos[i] for i in idx] for idx in preds]
trg_text = [[vocab.itos[i] for i in idx] for idx in y]
i = 1
list(zip(src_text[i], pred_text[i], trg_text[i]))


query = """
Entrepreneurs sought to capitalize on the wealth generated by the Gold Rush. Early winners were the banking industry, with the founding of Wells Fargo in 1852 and the Bank of California in 1864. Development of the Port of San Francisco and the establishment in 1869 of overland access to the eastern U.S. rail system via the newly completed Pacific Railroad (the construction of which the city only reluctantly helped support[41]) helped make the Bay Area a center for trade. Catering to the needs and tastes of the growing population, Levi Strauss opened a dry goods business and Domingo Ghirardelli began manufacturing chocolate. Immigrant laborers made the city a polyglot culture, with Chinese Railroad Workers, drawn to "Old Gold Mountain", creating the city's Chinatown quarter. In 1870, Asians made up 8% of the population.[42] The first cable cars carried San Franciscans up Clay Street in 1873. The city's sea of Victorian houses began to take shape, and civic leaders campaigned for a spacious public park, resulting in plans for Golden Gate Park. San Franciscans built schools, churches, theaters, and all the hallmarks of civic life. The Presidio developed into the most important American military installation on the Pacific coast."""


inp = tokenizer(query.replace("\n", " ").lower())
inp_t = torch.LongTensor([vocab.stoi[w] for w in inp[-10:]]).to("cuda:0").view(1, -1)
position_indices = torch.arange(pos_start, pos_start + 10, 
                                        device=devices[0],
                                        dtype=torch.long).repeat(1, 1).long()
inp_pos_t = torch.stack((inp_t, position_indices), dim=2)
feat = model.forward(inp_pos_t, Batch.make_std_mask(inp_t, pad_idx).to(devices[0]))
out = model.generator.forward(feat)
preds = torch.argmax(out, dim=2)
src_text = [[vocab.itos[i] for i in idx] for idx in inp_pos_t[:1, :, 0]]
pred_text = [[vocab.itos[i] for i in idx] for idx in preds]
list(zip(src_text[0], pred_text[0]))

l = valid_criterion(out.contiguous().view(-1, out.size(-1)), y.contiguous().view(-1))

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
  

