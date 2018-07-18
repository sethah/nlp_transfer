import numpy as np
import re

import torch
import torch.nn as nn

class MultiGPULossCompute(object):     
    "A multi-gpu loss compute and train function."
    def __init__(self, generator, criterion, devices):
        # Send out to different gpus.
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion, devices=devices)
        self.devices=devices
        
    def __call__(self, out, targets):
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

#        # Backprop loss to output of transformer
#        if self.opt is not None:
#            l.backward()
#            if self.clip > 0.0:
#              torch.nn.utils.clip_grad_norm_(params, self.clip)
#            self.opt.step()
#            self.opt.zero_grad()

        return l
  
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

def cyclical_lr(step_sz, min_lr=0.001, max_lr=1, mode='triangular', scale_func=None, scale_md='cycles', gamma=1.):
    """implements a cyclical learning rate policy (CLR).
    Notes: the learning rate of optimizer should be 1

    Parameters:
    ----------
    mode : str, optional
        one of {triangular, triangular2, exp_range}. 
    scale_md : str, optional
        {'cycles', 'iterations'}.
    gamma : float, optional
        constant in 'exp_range' scaling function: gamma**(cycle iterations)
    
    Examples:
    --------
    >>> # the learning rate of optimizer should be 1
    >>> optimizer = optim.SGD(model.parameters(), lr=1.)
    >>> step_size = 2*len(train_loader)
    >>> clr = cyclical_lr(step_size, min_lr=0.001, max_lr=0.005)
    >>> scheduler = lr_scheduler.LambdaLR(optimizer, [clr])
    >>> # some other operations
    >>> scheduler.step()
    >>> optimizer.step()
    """
    if scale_func == None:
        if mode == 'triangular':
            scale_fn = lambda x: 1.
            scale_mode = 'cycles'
        elif mode == 'triangular2':
            scale_fn = lambda x: 1 / (2.**(x - 1))
            scale_mode = 'cycles'
        elif mode == 'exp_range':
            scale_fn = lambda x: gamma**(x)
            scale_mode = 'iterations'
        else:
            raise ValueError(f'The {mode} is not valid value!')
    else:
        scale_fn = scale_func
        scale_mode = scale_md

    lr_lambda = lambda iters: min_lr + (max_lr - min_lr) * rel_val(iters, step_sz, scale_mode)

    def rel_val(iteration, stepsize, mode):
        cycle = np.floor(1 + iteration / (2 * stepsize))
        x = abs(iteration / stepsize - 2 * cycle + 1)
        if mode == 'cycles':
            return max(0, (1 - x)) * scale_fn(cycle)
        elif mode == 'iterations':
            return max(0, (1 - x)) * scale_fn(iteration)
        else:
            raise ValueError(f'The {scale_mode} is not valid value!')

    return lr_lambda
