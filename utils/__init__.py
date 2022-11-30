import os
import torch
from transformers import LongformerModel, LongformerTokenizerFast
from tqdm import tqdm
from copy import copy
import sklearn.metrics as metrics
import os
from utils.evaluate import load_answers, compute_metrics


class Logger:
    def __init__(self, path, sep=' ', end='\n'):
        self.path = os.path.join('log', path)
        self.sep = sep
        self.end = end
        if not os.path.exists('log'):
            os.mkdir('log')
        
    def log(self, *args):
        output_list = []
        for arg in args:
            output_list.append(str(arg))
        output_list.append(self.end)
        output = self.sep.join(output_list)
        with open(self.path, 'a', encoding = 'utf-8') as file:
            file.write(output)

def init_logger(path, sep=' ', end='\n'):
    logger = Logger(path, sep, end)
    return logger



class Tokenizer:
    def __init__(self, model_root):
        self.tokenizer = LongformerTokenizerFast.from_pretrained(os.path.join(model_root, 'LongformerTokenizer'))
        self.tokenizer.special_tokens_map.update({'yes_token':'<yes>','no_token':'<no>','na_token':'<na>'})
        self.tokenizer.add_tokens(
            ['<s>','<h1>','<h2>','<h3>','<h4>','<p>','<li>','<tr>',
            '</s>','</h1>','</h2>','</h3>','</h4>','</p>','</li>','</tr>',]
        )
        self.tokenizer.add_tokens(
            ['<yes>','<no>','<na>']
        )
        self.tokenizer.elem_tokens = self.tokenizer('<h1> <h2> <h3> <h4> <p> <li> <tr>').input_ids[1:-1]

    def __call__(self, input_str):
        return self.tokenizer(input_str, return_tensors = 'pt', padding='longest').input_ids[0,1:-1]
    
    def decode(self, tokens):
        return self.tokenizer.decode(tokens)



def input_to_batch(inputs, batch_size = 4, distributed = False):
    if distributed == False:
        batches = torch.utils.data.DataLoader(inputs, batch_size = batch_size, shuffle = True)
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(inputs)
        batches = torch.utils.data.DataLoader(inputs, batch_size=batch_size, shuffle = (sampler is None), sampler=sampler, pin_memory=True)
    return batches


class ReIndexer:
    def __init__(self):
        '''
        a class to make re-index in tensor easier
        code example:
            from utils import ReIndexer
            indexer = ReIndexer()
            hiddens = torch.arange(40).reshape(2, 10, 2)
            mask = torch.rand(2, 10) > 0.8
            mask = mask.float()
            indexer.set_index(mask)
            new_hiddens, new_mask = indexer.re_index(hiddens)
        '''
        self.index = None
        self.mask = None
        self.mask_r = None

    def set_index(self, mask: torch.FloatTensor):
        mask = mask.float()
        sign = torch.ones_like(mask).cumsum(-1).flip(-1)
        sign = torch.where(mask.bool(), sign, 0)
        _, index = sign.sort(-1, descending = True)
        self.index = index
        self.mask = mask.detach().cpu()
        self.mask_r, _ = self.mask.sort(-1, descending = True)

    def re_index(self, tensor):
        if tensor.dim() > self.index.dim():
            index = self.index.unsqueeze(-1).repeat(1,1,tensor.shape[2])
        else:
            index = self.index
        sorted_tensor = torch.gather(tensor, 1, index)
        return sorted_tensor
    
    def recover_index(self, tensor):
        if tensor.dim() > self.index.dim():
            index = self.index.unsqueeze(-1).repeat(1,1,tensor.shape[2])
        else:
            index = self.index
        recovered_tensor = torch.scatter(tensor, 1, index, tensor)
        return recovered_tensor

    def get_max_head_num(self):
        return self.mask_r.count_nonzero(-1).max()
    
    def get_mask_r(self):
        return self.mask_r[:, :self.get_max_head_num()].bool()


def get_level(text):
    return text[0][0] - 50264

class TxtNode:
    def __init__(self, text, tokenizer):
        self.tokenizer = tokenizer
        self.text = text
        self.level = get_level(text)
        self.children = []
        self.parent = None

    def get_nodes_list(self):
        if self.children == []:
            return [self]
        else:
            nodes = [self]
            for child in self.children:
                nodes += child.get_nodes_list()
            return nodes


    def __str__(self, indent = 0):
        string = self.tokenizer.tokenizer.decode([i[0] for i in self.text][:10])
        string = ' ' * indent + string
        if self.children == []:
            return string
        else:
            return '\n'.join([string] + [node.__str__(indent + 4) for node in self.children])

    __repr__ = __str__


    def __len__(self):
        nodes = self.get_nodes_list()
        return sum(len(i.text) for i in nodes)

        
    def copy(self): # deep copy 
        newnode = TxtNode(self.text, self.tokenizer)
        for child in self.children:
            newnode.children.append(child.copy())
        for child in newnode.children:
            child.parent = newnode
        return newnode


def dynamic_padding(attn_masks, *args):
    input_lengths = attn_masks.count_nonzero(dim = 1)
    length = input_lengths.max()
    return [attn_masks[:, :length]] + [i[:, :length] for i in args]

    
def analyze_binary_classification(label_answer_sentence, pred_answer_sentence, prec_weight = 1):
    fpr, tpr, thresholds = metrics.roc_curve(label_answer_sentence, pred_answer_sentence, pos_label = 1)
    auc = metrics.auc(fpr, tpr)
    p, r, thresholds = metrics.precision_recall_curve(label_answer_sentence, pred_answer_sentence, pos_label = 1)
    f1_score = torch.tensor((1 + prec_weight) * p * r / (p + prec_weight * r + 0.00001))
    return (f1_score.max(), auc, thresholds[f1_score.argmax()], p[f1_score.argmax()], r[f1_score.argmax()])

def to_numpy(*args):
    return [arg.detach().cpu().tolist() for arg in args]


def compare(pred_file='outputs/output', ref_file='../condqa_files/data/dev.json',compare_file='outputs/compare.out'):
  qid2predictions = load_answers(pred_file)
  qid2references = load_answers(ref_file)
  with open(compare_file, 'w') as file:
      pass
  f1s = 0
  for qid in qid2references.keys():
    em, conditional_em, f1, conditional_f1 = compute_metrics(
    qid2predictions[qid], qid2references[qid])
    f1s += f1
    with open(compare_file, 'a') as file:
      file.write(str([qid, f1, [i[0] for i in qid2predictions[qid]], [i[0] for i in qid2references[qid]]]) + '\n')
  print(f1s)
    
