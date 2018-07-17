sys.path.append("/home/cdsw/pytorch-openai-transformer-lm/")
import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from sklearn.metrics import accuracy_score

from datasets import _rocstories

def rocstories(data_dir, pred_path, log_path):
    preds = pd.read_csv(pred_path, delimiter='\t')['prediction'].values.tolist()
    _, _, _, labels = _rocstories(os.path.join(data_dir, 'cloze_test_test__spring2016 - cloze_test_ALL_test.csv'))
    test_accuracy = accuracy_score(labels, preds)*100.
    logs = [json.loads(line) for line in open(log_path)][1:]
    best_validation_index = np.argmax([log['va_acc'] for log in logs])
    valid_accuracy = logs[best_validation_index]['va_acc']
    print('ROCStories Valid Accuracy: %.2f'%(valid_accuracy))
    print('ROCStories Test Accuracy:  %.2f'%(test_accuracy))

from datasets import rocstories
from model_pytorch import DoubleHeadModel, load_openai_pretrained_model
from opt import OpenAIAdam
from text_utils import TextEncoder
from hfutils import (encode_dataset, iter_data,
                   ResultLogger, make_path)

class LossCompute:
    "A Loss compute and train function."

    def __init__(self, lm_criterion, clf_criterion, lm_coef, opt=None):
        self.lm_criterion = lm_criterion
        self.clf_criterion = clf_criterion
        self.lm_coef = lm_coef
        self.opt = opt

    def __call__(self, X, Y, M, clf_logits, lm_logits=None, only_return_losses=False):
        # Language modeling loss
        if lm_logits is not None:
            x_shifted = X[:, :, 1:, 0].contiguous().view(-1)  # Shape: 252
            M = M.view(-1, M.size(2))
            lm_losses = self.lm_criterion(lm_logits, x_shifted)
            lm_losses = lm_losses.view(X.size(0) * X.size(1), X.size(2) - 1)
            lm_losses = lm_losses * M[:, 1:]
            lm_losses = lm_losses.sum(1) / torch.sum(M[:, 1:], 1)
        # Classification loss
        clf_losses = self.clf_criterion(clf_logits, Y)
        if only_return_losses:
            return (clf_losses, lm_losses) if lm_logits is not None else clf_losses

        if self.lm_coef > 0 and lm_logits is not None:
            train_loss = clf_losses.sum() + self.lm_coef * lm_losses.sum()
        else:
            train_loss = clf_losses.sum()
        train_loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.zero_grad()
        return train_loss.item()


def transform_roc(X1, X2, X3):
    n_batch = len(X1)
    xmb = np.zeros((n_batch, 2, n_ctx, 2), dtype=np.int32)
    mmb = np.zeros((n_batch, 2, n_ctx), dtype=np.float32)
    start = encoder['_start_']
    delimiter = encoder['_delimiter_']
    for i, (x1, x2, x3), in enumerate(zip(X1, X2, X3)):
        x12 = [start] + x1[:max_len] + [delimiter] + x2[:max_len] + [clf_token]
        x13 = [start] + x1[:max_len] + [delimiter] + x3[:max_len] + [clf_token]
        l12 = len(x12)
        l13 = len(x13)
        xmb[i, 0, :l12, 0] = x12
        xmb[i, 1, :l13, 0] = x13
        mmb[i, 0, :l12] = 1
        mmb[i, 1, :l13] = 1
    xmb[:, :, :, 1] = np.arange(n_vocab + n_special, n_vocab + n_special + n_ctx)
    return xmb, mmb


def iter_apply(Xs, Ms, Ys):
    # fns = [lambda x: np.concatenate(x, 0), lambda x: float(np.sum(x))]
    logits = []
    cost = 0
    with torch.no_grad():
        dh_model.eval()
        for xmb, mmb, ymb in iter_data(Xs, Ms, Ys, n_batch=n_batch_train, truncate=False, verbose=True):
            n = len(xmb)
            XMB = torch.tensor(xmb, dtype=torch.long).to(device)
            YMB = torch.tensor(ymb, dtype=torch.long).to(device)
            MMB = torch.tensor(mmb).to(device)
            _, clf_logits = dh_model(XMB)
            clf_logits *= n
            clf_losses = compute_loss_fct(XMB, YMB, MMB, clf_logits, only_return_losses=True)
            clf_losses *= n
            logits.append(clf_logits.to("cpu").numpy())
            cost += clf_losses.sum().item()
        logits = np.concatenate(logits, 0)
    return logits, cost


def iter_predict(Xs, Ms):
    logits = []
    with torch.no_grad():
        dh_model.eval()
        for xmb, mmb in iter_data(Xs, Ms, n_batch=n_batch_train, truncate=False, verbose=True):
            n = len(xmb)
            XMB = torch.tensor(xmb, dtype=torch.long).to(device)
            MMB = torch.tensor(mmb).to(device)
            _, clf_logits = dh_model(XMB)
            logits.append(clf_logits.to("cpu").numpy())
    logits = np.concatenate(logits, 0)
    return logits


def log(save_dir, desc):
    global best_score
    print("Logging")
    tr_logits, tr_cost = iter_apply(trX[:n_valid], trM[:n_valid], trY[:n_valid])
    va_logits, va_cost = iter_apply(vaX, vaM, vaY)
    tr_cost = tr_cost / len(trY[:n_valid])
    va_cost = va_cost / n_valid
    tr_acc = accuracy_score(trY[:n_valid], np.argmax(tr_logits, 1)) * 100.
    va_acc = accuracy_score(vaY, np.argmax(va_logits, 1)) * 100.
    logger.log(n_epochs=n_epochs, n_updates=n_updates, tr_cost=tr_cost, va_cost=va_cost, tr_acc=tr_acc, va_acc=va_acc)
    print('%d %d %.3f %.3f %.2f %.2f' % (n_epochs, n_updates, tr_cost, va_cost, tr_acc, va_acc))
    if submit:
        score = va_acc
        if score > best_score:
            best_score = score
            path = os.path.join(save_dir, desc, 'best_params')
            torch.save(dh_model.state_dict(), make_path(path))


def predict(dataset, submission_dir):
    filename = filenames[dataset]
    pred_fn = pred_fns[dataset]
    label_decoder = label_decoders[dataset]
    predictions = pred_fn(iter_predict(teX, teM))
    if label_decoder is not None:
        predictions = [label_decoder[prediction] for prediction in predictions]
    path = os.path.join(submission_dir, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('{}\t{}\n'.format('index', 'prediction'))
        for i, prediction in enumerate(predictions):
            f.write('{}\t{}\n'.format(i, prediction))


def run_epoch():
    for xmb, mmb, ymb in iter_data(*shuffle(trX, trM, trYt, random_state=np.random),
                                   n_batch=n_batch_train, truncate=True, verbose=True):
        global n_updates
        dh_model.train()
        XMB = torch.tensor(xmb, dtype=torch.long).to(device)
        YMB = torch.tensor(ymb, dtype=torch.long).to(device)
        MMB = torch.tensor(mmb).to(device)
        lm_logits, clf_logits = dh_model(XMB)
        compute_loss_fct(XMB, YMB, MMB, clf_logits, lm_logits)
        n_updates += 1
        if n_updates in [1000, 2000, 4000, 8000, 16000, 32000] and n_epochs == 0:
            log(save_dir, desc)


argmax = lambda x: np.argmax(x, 1)

pred_fns = {
    'rocstories': argmax,
}

filenames = {
    'rocstories': 'ROCStories.tsv',
}

label_decoders = {
    'rocstories': None,
}

parser = argparse.ArgumentParser()
parser.add_argument('--desc', type=str, help="Description")
parser.add_argument('--dataset', type=str)
parser.add_argument('--log_dir', type=str, default='log/')
parser.add_argument('--save_dir', type=str, default='save/')
parser.add_argument('--data_dir', type=str, default='data/')
parser.add_argument('--submission_dir', type=str, default='submission/')
parser.add_argument('--submit', action='store_true')
parser.add_argument('--analysis', action='store_true')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--n_iter', type=int, default=3)
parser.add_argument('--n_batch', type=int, default=8)
parser.add_argument('--max_grad_norm', type=int, default=1)
parser.add_argument('--lr', type=float, default=6.25e-5)
parser.add_argument('--lr_warmup', type=float, default=0.002)
parser.add_argument('--n_ctx', type=int, default=512)
parser.add_argument('--n_embd', type=int, default=768)
parser.add_argument('--n_head', type=int, default=12)
parser.add_argument('--n_layer', type=int, default=12)
parser.add_argument('--embd_pdrop', type=float, default=0.1)
parser.add_argument('--attn_pdrop', type=float, default=0.1)
parser.add_argument('--resid_pdrop', type=float, default=0.1)
parser.add_argument('--clf_pdrop', type=float, default=0.1)
parser.add_argument('--l2', type=float, default=0.01)
parser.add_argument('--vector_l2', action='store_true')
parser.add_argument('--opt', type=str, default='adam')
parser.add_argument('--afn', type=str, default='gelu')
parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
parser.add_argument('--encoder_path', type=str, default='model/encoder_bpe_40000.json')
parser.add_argument('--bpe_path', type=str, default='model/vocab_40000.bpe')
parser.add_argument('--n_transfer', type=int, default=12)
parser.add_argument('--lm_coef', type=float, default=0.5)
parser.add_argument('--b1', type=float, default=0.9)
parser.add_argument('--b2', type=float, default=0.999)
parser.add_argument('--e', type=float, default=1e-8)
parser.add_argument('--n_valid', type=int, default=374)

args = parser.parse_args(args=["--bpe_path", "/home/cdsw/pytorch-openai-transformer-lm/model/vocab_40000.bpe",
                              "--encoder_path", "/home/cdsw/pytorch-openai-transformer-lm/model/encoder_bpe_40000.json",
                              "--data_dir", "/home/cdsw/data/",
                              "--"])

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Constants
submit = args.submit
dataset = args.dataset
n_ctx = args.n_ctx
save_dir = args.save_dir
desc = args.desc
data_dir = args.data_dir
log_dir = args.log_dir
submission_dir = args.submission_dir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("device", device, "n_gpu", n_gpu)

logger = ResultLogger(path=os.path.join(log_dir, '{}.jsonl'.format(desc)), **args.__dict__)
text_encoder = TextEncoder(args.encoder_path, args.bpe_path)
encoder = text_encoder.encoder
n_vocab = len(text_encoder.encoder)

print("Encoding dataset...")
((trX1, trX2, trX3, trY),
 (vaX1, vaX2, vaX3, vaY),
 (teX1, teX2, teX3)) = encode_dataset(*rocstories(data_dir, n_valid=args.n_valid),
                                      encoder=text_encoder)

encoder['_start_'] = len(encoder)
encoder['_delimiter_'] = len(encoder)
encoder['_classify_'] = len(encoder)
clf_token = encoder['_classify_']
n_special = 3
max_len = n_ctx // 2 - 2
n_ctx = min(max(
    [len(x1[:max_len]) + max(len(x2[:max_len]),
                             len(x3[:max_len])) for x1, x2, x3 in zip(trX1, trX2, trX3)]
    + [len(x1[:max_len]) + max(len(x2[:max_len]),
                               len(x3[:max_len])) for x1, x2, x3 in zip(vaX1, vaX2, vaX3)]
    + [len(x1[:max_len]) + max(len(x2[:max_len]),
                               len(x3[:max_len])) for x1, x2, x3 in zip(teX1, teX2, teX3)]
    ) + 3, n_ctx)

vocab_size = n_vocab + n_special + n_ctx
trX, trM = transform_roc(trX1, trX2, trX3)
vaX, vaM = transform_roc(vaX1, vaX2, vaX3)
if submit:
    teX, teM = transform_roc(teX1, teX2, teX3)
    
trYt = trY
n_batch_train = args.n_batch * max(n_gpu, 1)
for xmb, mmb, ymb in iter_data(*shuffle(trX, trM, trYt, random_state=np.random),
                                   n_batch=n_batch_train, truncate=True, verbose=True):
  print(xmb.shape)
  break
  
dh_model = DoubleHeadModel(args, clf_token, vocab_size, n_ctx)

criterion = nn.CrossEntropyLoss(reduce=False)
model_opt = OpenAIAdam(dh_model.parameters(),
                       lr=args.lr,
                       schedule=args.lr_schedule,
                       warmup=args.lr_warmup,
                       t_total=n_updates_total,
                       b1=args.b1,
                       b2=args.b2,
                       e=args.e,
                       l2=args.l2,
                       vector_l2=args.vector_l2,
                       max_grad_norm=args.max_grad_norm)
compute_loss_fct = LossCompute(criterion,
                               criterion,
                               args.lm_coef,
                               model_opt)
load_openai_pretrained_model(dh_model.transformer, n_ctx=n_ctx, n_special=n_special,
                            path_names="/home/cdsw/pytorch-openai-transformer-lm/",
                            path="/home/cdsw/finetune-transformer-lm/model/")
