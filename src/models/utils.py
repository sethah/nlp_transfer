import numpy as np
import re

import torch
import torch.nn as nn

class MultiGPULossCompute(object):     
    "A multi-gpu loss compute and train function."
    def __init__(self, generator, criterion, devices, opt=None, clip=0.0):
        # Send out to different gpus.
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion, devices=devices)
        self.opt = opt
        self.devices=devices
        self.clip = clip
        
    def __call__(self, out, targets, params):
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
            if self.clip > 0.0:
              torch.nn.utils.clip_grad_norm_(params, self.clip)
            self.opt.step()
            self.opt.zero_grad()

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

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src = batch.text.transpose(0, 1)
    return Batch(batch.text.transpose(0, 1), batch.target.transpose(0, 1), pad_idx)
  
def text_standardize(text):
    """
    fixes some issues the spacy tokenizer had on books corpus
    also does some whitespace standardization
    """
    text = text.replace('—', '-')
    text = text.replace('–', '-')
    text = text.replace('―', '-')
    text = text.replace('…', '...')
    text = text.replace('´', "'")
    text = re.sub(r'''(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)''', r' \1 ', text)
    text = re.sub(r'\s*\n\s*', ' \n ', text)
    text = re.sub(r'[^\S\n]+', ' ', text)
    return text.strip()

