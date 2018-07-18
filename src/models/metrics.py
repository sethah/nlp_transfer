import numpy as np
from ignite.metrics import Metric

class Perplexity(Metric):
  
    def __init__(self, loss_compute):
        super(Perplexity, self).__init__()
        self._loss_compute = loss_compute

    def reset(self):
        self._total_nll = 0
        self._num_examples = 0

    def update(self, output):
        y_pred, y, ntokens = output
        batch_size = y.shape[0]
        nll = self._loss_compute(y_pred, y).item()
        self._total_nll += nll
        self._num_examples += ntokens

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('must have at least one example before it can be computed')
        return np.exp(self._total_nll / self._num_examples)
