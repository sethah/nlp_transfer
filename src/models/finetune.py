import numpy as np
import math, copy, time
import os
import itertools
import spacy
import json
from collections import Counter, defaultdict
import ftfy
import re
import pickle
import argparse
import sys

# pytorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data, datasets
from torchtext.vocab import Vocab
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers.checkpoint import ModelCheckpoint

# local packages
from models.modules import *
from models.load_pretrain import load_model, load_vocab
import models.utils
from models.text_utils import TextEncoder, AttentionIterator, Batch
from models.metrics import Perplexity
from models.opt import OpenAIAdam, GradualUnfreezingOptimizer

def unfreeze(param_group):
  for p in param_group:
    p.requires_grad = True
      
def freeze(param_group):
  for p in param_group:
    p.requires_grad = False
    
   
"""
PYTHONPATH=/home/cdsw/nlp_transfer/src/ python3 models/finetune.py \
--num_gpus 4 \
--batch_size 128 \
--bptt_len 70 \
--num_epochs 7 \
--checkpoint_dir /home/cdsw/nlp_transfer/models/checkpoints/ \
--check_prefix test5 \
--data_path /home/cdsw/nlp_transfer/data/processed/wikitext2_bpe/ \
--models_path /home/cdsw/nlp_transfer/models/openai-transformer-lm/
"""

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--num_gpus', type=int, default=-1)
  parser.add_argument('--batch_size', type=int)
  parser.add_argument('--checkpoint_dir', type=str)
  parser.add_argument('--data_path', type=str)
  parser.add_argument('--pad_token', type=str, default="<pad>")
  parser.add_argument('--phases', type=str, default="train,test,valid")
  parser.add_argument('--bptt_len', type=int)
  parser.add_argument('--num_epochs', type=int)
  parser.add_argument('--models_path', type=str)
  parser.add_argument('--check_prefix', type=str, default="chk")
  args = parser.parse_args()
  
#  if os.path.exists(os.path.join(args.checkpoint_dir, args.check_prefix)):
#    raise Exception("checkpoint path already exists")
#  else:
#    os.mkdir(os.path.join(args.checkpoint_dir, args.check_prefix))

  num_gpus = args.num_gpus if args.num_gpus > -1 else torch.cuda.device_count()
  devices = [torch.device("cuda", i) for i in range(num_gpus)]
  device_ids = [d.index for d in devices]
  
  itos = pickle.load(open(os.path.join(args.data_path, "vocab.pkl"), 'rb'))
  stoi = {s: i for i, s in enumerate(itos)}
  vocab = Vocab(Counter({}), specials=[])
  vocab.stoi = stoi
  vocab.itos = itos

  TEXT = data.Field(lower=True)
  TEXT.vocab = vocab

  phases = args.phases.split(",")
  idx_tokens = {phase: np.load(os.path.join(args.data_path, f"{phase}.npy")) for phase in phases}

  pos_start = len(vocab)
  pad_idx = vocab.stoi[args.pad_token]
  iters = {}
  for phase, idx_array in idx_tokens.items():
    fields = [('text', TEXT)]
    ex = data.Example()
    setattr(ex, 'text', [vocab.itos[i] for i in idx_array])
    ds = data.Dataset([ex], fields)
    it = data.BPTTIterator(ds, args.batch_size, args.bptt_len, repeat=False, shuffle=phase=='train', device=device_ids[0])
    iters[phase] = AttentionIterator(it, pos_start, pad_idx)

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
  
  special_embeds = len(vocab) - 40478
  if args.resume is not None:
    d = torch.load(args.resume)
    model = d['model']
    model_opt = d['opt']
  else:
    embeds, encoder = load_model(special_embeds=special_embeds, weights_path=args.models_path)
    generator = Generator(768, len(vocab), 0.3)
    model = EncoderOnly(encoder, embeds, generator).to(devices[0])
    
    initial_lr = 0.01
    layer_groups = [{'name': 'generator', 'params': list(generator.parameters()), 'initial_lr': initial_lr},
                    {'name': 'encoder3', 'params': [plist for l in encoder.layers[::-1][0:3] for plist in list(l.parameters())], 'initial_lr': initial_lr / 2},
                    {'name': 'encoder2', 'params': [plist for l in encoder.layers[::-1][3:6] for plist in list(l.parameters())], 'initial_lr': initial_lr / 4},
                    {'name': 'encoder1', 'params': [plist for l in encoder.layers[::-1][6:9] for plist in list(l.parameters())], 'initial_lr': initial_lr / 8},
                    {'name': 'encoder0', 'params': [plist for l in encoder.layers[::-1][9:12] for plist in list(l.parameters())], 'initial_lr': initial_lr / 16},
                    {'name': 'embeds', 'params': list(embeds.parameters()), 'initial_lr': initial_lr / 32}]
    initial_opt_params = layer_groups[:1]
    model_opt = OpenAIAdam(initial_opt_params,
                           lr=initial_lr,
                           schedule='warmup_linear',
                           warmup=0.005,
                           t_total=len(iters['train'].iterator) * 1,
                           b1=0.8,
                           b2=0.99,
                           e=1e-8,
                           l2=0.01,
                           vector_l2='store_true',
                           max_grad_norm=0.25)

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

  
  model_par = nn.DataParallel(model, device_ids=[d.index for d in devices])
  criterion = nn.NLLLoss(reduce=False, size_average=False).to(devices[0])
  valid_criterion = nn.NLLLoss(reduce=False, size_average=False).to(devices[0])

  trainer = Engine(training_update_function)
  evaluator = Engine(inference)

  # measure perplexity
  metric = Perplexity(models.utils.MultiGPULossCompute(model.generator, valid_criterion, device_ids))
  metric.attach(evaluator, "ppl")

  results = {'best_valid': 100000}

  @trainer.on(Events.ITERATION_COMPLETED)
  def log_training_loss(trainer):
      if (trainer.state.iteration + 1) % 20 == 0:
        print("Epoch[{}] Iteration[{}] Loss: {:.2f}".format(trainer.state.epoch, 
                                                            trainer.state.iteration + 1,
                                                            trainer.state.output))

  @trainer.on(Events.EPOCH_COMPLETED)
  def log_validation(trainer):
      evaluator.run(iters['valid'])
      loss = evaluator.state.metrics['ppl']
      if (loss < results['best_valid']):
        results['best_valid'] = loss
        torch.save({'model': model, 'opt': model_opt}, os.path.join(args.checkpoint_dir, args.check_prefix, "best_loss"))
      print("Epoch[{}]  Valid loss: {:.2f}".format(trainer.state.epoch, loss))
      
  @trainer.on(Events.EPOCH_COMPLETED)
  def unfreeze_layers(trainer):
    unfreezer.step_epoch()

  @trainer.on(Events.EPOCH_STARTED)
  def log_lr(trainer):
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    print(f'Now training {len(trainable_params)} parameters')
    for param_group in model_opt.param_groups:
      print(param_group['name'], param_group['lr'])

  @trainer.on(Events.ITERATION_COMPLETED)
  def lr_step(trainer):
      if (trainer.state.iteration + 1) % 1 == 0:
        unfreezer.step_iteration()
      if (trainer.state.iteration + 1) % 1 == 0:
        print("Epoch[%d] Iteration[%d] New learning rates: %s" % (trainer.state.epoch,
                                                                 trainer.state.iteration + 1,
                                                                 unfreezer.get_lr()))

  def get_iter_scheduler(opt):
    clr = models.utils.cyclical_lr(len(iters['train'].iterator) // 2, min_lr=0.1, max_lr=1, mode='triangular')
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(model_opt, [clr] * len(opt.param_groups))
    return lr_scheduler
  def get_epoch_scheduler(opt, **kwargs):
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(model_opt, gamma=0.95, **kwargs)
    return lr_scheduler
  unfreezer = GradualUnfreezingOptimizer(model_opt, layer_groups, epoch=0, get_iter_scheduler=get_iter_scheduler)

  checkpointer = ModelCheckpoint(args.checkpoint_dir, args.check_prefix, save_interval=1,
                                create_dir=True, require_empty=False)
  trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model, 'opt': model_opt})

  for group in layer_groups:
    freeze(group['params'])
  unfreeze(layer_groups[0]['params'])
  train_state = trainer.run(iters['train'], max_epochs=args.num_epochs)


  eval_state = evaluator.run(iters['test'])
  print("Test set perplexity: %0.2f" % eval_state.metrics['ppl'])
