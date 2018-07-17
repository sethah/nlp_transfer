import numpy as np
import os, sys
import json

# pytorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data, datasets
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers.checkpoint import ModelCheckpoint

# local packages
from models.models import *
import models.utils

def load_model(special_embeds=0, weights_path="model/"):

  # params is just one big array split in 10 files
  params = [np.load(os.path.join(weights_path, "params_%d.npy" % i)) for i in range(10)]
  shapes = json.load(open(os.path.join(weights_path, 'params_shapes.json')))
  offsets = np.cumsum([np.prod(shape) for shape in shapes])

  arrays = np.split(np.concatenate(params, 0), offsets)[:-1]
  arrays = [param.reshape(shape) for param, shape in zip(arrays, shapes)]
  arrays = [arr.squeeze() for arr in arrays]

  embeds_np = arrays[1]
  vocab_len = embeds_np.shape[0]
  pos_embeds_np = arrays[0]
  pos_embeds_np.shape
  embed_dim = embeds_np.shape[1]
  
  if special_embeds == 0:
    embeds_np = np.concatenate([embeds_np, pos_embeds_np], 0)
  else:
    extra_embeds = (np.random.randn(special_embeds, embed_dim) * 0.02).astype(np.float32)
    embeds_np = np.concatenate([embeds_np, extra_embeds, pos_embeds_np], 0)


  # model parameters
  h = 12
  d_model = 768
  d_ff = 3072
  dropout = 0.1
  n_layers = 12

  attn = MultiHeadedAttention(h, d_model)
  ff = PositionwiseFeedForward(d_model, d_ff, dropout)
  position = PositionalEncoding(d_model, dropout)
  encoder = Encoder(EncoderLayer(d_model, attn, ff, dropout), n_layers)
  embeds = Embeddings(d_model, vocab_len)
  #embeds.lut.weight.requires_grad = False
  #embeds.lut.weight.copy_(vocab.vectors)
#  pos_embeds = nn.Sequential(embeds, position)
  generator = Generator(d_model, vocab_len)
  #criterion = LabelSmoothing(size=vocab_len, padding_idx=pad_idx, smoothing=0.1).to(devices[0])

#  shapes = dict(list(encoder.named_parameters()))
#  for n, p in encoder.named_parameters():
#    print(n, p.data.shape)
  embeds.lut.weight.data = torch.from_numpy(embeds_np)

  j = 2
  for enc_layer in encoder.layers:
    attn = enc_layer.self_attn
    attn_qkv_np = arrays[j]
    attn_qkv_bias_np = arrays[j + 1]
    attn_proj_np = arrays[j + 2]
    attn_proj_bias_np = arrays[j + 3]
    attn.linears[0].weight.data = torch.from_numpy(attn_qkv_np[:, :d_model].T)
    attn.linears[0].bias.data = torch.from_numpy(attn_qkv_bias_np[:d_model])
    attn.linears[1].weight.data = torch.from_numpy(attn_qkv_np[:, d_model:2 * d_model].T)
    attn.linears[1].bias.data = torch.from_numpy(attn_qkv_bias_np[d_model:2 * d_model])
    attn.linears[2].weight.data = torch.from_numpy(attn_qkv_np[:, 2 * d_model:3 * d_model].T)
    attn.linears[2].bias.data = torch.from_numpy(attn_qkv_bias_np[2 * d_model:3 * d_model])
    attn.linears[3].weight.data = torch.from_numpy(attn_proj_np.T)
    attn.linears[3].bias.data = torch.from_numpy(attn_proj_bias_np)

    ln1 = enc_layer.sublayer[0].norm
    ln2 = enc_layer.sublayer[1].norm
    ln1.a_2.data = torch.from_numpy(arrays[j + 4].T)
    ln1.b_2.data = torch.from_numpy(arrays[j + 5].T)

    ff = enc_layer.feed_forward
    ff.w_1.weight.data = torch.from_numpy(arrays[j + 6].T)
    ff.w_1.bias.data = torch.from_numpy(arrays[j + 7])
    ff.w_2.weight.data = torch.from_numpy(arrays[j + 8].T)
    ff.w_2.bias.data = torch.from_numpy(arrays[j + 9])

    ln2.a_2.data = torch.from_numpy(arrays[j + 10].T)
    ln2.b_2.data = torch.from_numpy(arrays[j + 11].T)
    j += 12
  
#  for n, p in encoder.named_parameters():
#    print(n, p.data.shape)
    
  return embeds, encoder, generator#, shapes, dict(list(encoder.named_parameters()))