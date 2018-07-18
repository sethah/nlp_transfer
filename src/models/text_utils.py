import re
import ftfy
import json
import spacy

from tqdm import tqdm

import torch
import torchtext.data

def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    word is represented as tuple of symbols (symbols being variable-length strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

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

class TextEncoder(object):
    """
    mostly a wrapper for a public python bpe tokenizer
    """

    def __init__(self, encoder_path, bpe_path):
        self.nlp = spacy.load('en', disable=['parser', 'tagger', 'ner', 'textcat'])
        self.encoder = json.load(open(encoder_path))
        self.decoder = {v:k for k,v in self.encoder.items()}
        merges = open(bpe_path, encoding='utf-8').read().split('\n')[1:-1]
        merges = [tuple(merge.split()) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}

    def bpe(self, token):
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        if token in self.cache:
            return self.cache[token]
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        if word == '\n  </w>':
            word = '\n</w>'
        self.cache[token] = word
        return word

    def encode(self, texts, verbose=True):
        texts_tokens = []
        if verbose:
            for text in tqdm(texts, ncols=80, leave=False):
                text = self.nlp(text_standardize(ftfy.fix_text(text)))
                text_tokens = []
                for token in text:
                    text_tokens.extend([self.encoder.get(t, 0) for t in self.bpe(token.text.lower()).split(' ')])
                texts_tokens.append(text_tokens)
        else:
            for text in texts:
                text = self.nlp(text_standardize(ftfy.fix_text(text)))
                text_tokens = []
                for token in text:
                    text_tokens.extend([self.encoder.get(t, 0) for t in self.bpe(token.text.lower()).split(' ')])
                texts_tokens.append(text_tokens)
        return texts_tokens
      
class AttentionIterator(object):
  
    def __init__(self, iterator, pos_start_index, mask_value=0):
      self.iterator = iterator
      self.mask_value = mask_value
      self.pos_start_index = pos_start_index
    
    def __iter__(self):
      for batch in self.iterator:
        masked = AttentionIterator._mask(batch, self.mask_value)
        batch_size, seq_len = masked.src.shape
        position_indices = torch.arange(self.pos_start_index, self.pos_start_index + seq_len, 
                                        device=masked.src.device,
                                        dtype=torch.long).repeat(batch_size, 1).long()
        masked.src = torch.stack((masked.src, position_indices), dim=2)
        yield masked
    
    @staticmethod
    def _mask(batch, mask_value=0):
        "Fix order in torchtext to match ours"
        src = batch.text.transpose(0, 1)
        batch_size, seq_len = src.shape
        return Batch(src, batch.target.transpose(0, 1), mask_value)
  
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
        tgt_mask = tgt_mask & Batch.subsequent_mask(tgt.size(-1)).type_as(tgt_mask)
        return tgt_mask
      
    @staticmethod
    def subsequent_mask(size):
        "Mask out subsequent positions."
        attn_shape = (size, size)
        subsequent_mask = torch.triu(torch.ones((size, size)), diagonal=1).type(torch.uint8).view(1, size, size)
        return subsequent_mask == 0
  
