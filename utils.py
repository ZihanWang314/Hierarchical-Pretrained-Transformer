import os
import torch
from transformers import LongformerModel, LongformerTokenizerFast
import json
from tqdm import tqdm
from torch import nn
from copy import copy
import numpy as np
import sklearn.metrics as metrics
import os
from random import shuffle
from evaluate import evaluate
import argparse


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

def check_and_transform_long_example(qa):
    return_qas = []
    LENGTH = 150
    STEP = 120
    subdoc_index = [0]
    while subdoc_index[-1] + LENGTH < len(qa['document']):
        subdoc_index.append(subdoc_index[-1] + STEP)
    for start_index in subdoc_index:
        return_qa = copy(qa)
        return_qa['document'] = qa['document'][start_index: start_index + LENGTH]
        return_qas.append(return_qa)
    return return_qas

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


def convert_examples_to_inputs(examples, tokenizer):
    examples_short = []
    for qa in tqdm(examples, desc = 'checking long example..'):
        examples_short += check_and_transform_long_example(qa)
    inputs = []

    for qa in tqdm(examples_short, desc = 'converting examples to inputs..'):
        input_head = tokenizer('Title: ' + qa['title'] + ' Document: ')
        input_tail = tokenizer(' Question: ' + qa['question'] + ' Scenario: ' + qa['scenario']) 

        input_text = [input_head]
        global_mask = [torch.ones(input_head.shape[0])]
        index_HTMLelements = []
        label_HTMLelements = []
        label_answer_span = []
        mask_label_condition = [torch.zeros(len(input_head), 5, 2)]

        if {'yes', 'no'} & {i[0] for i in qa['answers']}:
            yesno = True
            yes_evidences = set()
            no_evidences = set()
            for answer in qa['answers']:
                if answer[0] == 'yes':
                    yes_evidences.update(set(answer[1]))
                if answer[0] == 'no':
                    no_evidences.update(set(answer[1]))
            for evidence in qa['evidences']:
                if evidence not in yes_evidences and evidence not in no_evidences:
                    if 'yes' in {i[0] for i in qa['answers']}:
                        yes_evidences.add(evidence)
                    if 'no' in {i[0] for i in qa['answers']}:
                        no_evidences.add(evidence)

        else:
            yesno = False
        
        for sentence in qa['document']:
            tokens = sentence['tokens']
            global_mask_sentence = torch.zeros(tokens.shape[0])
            global_mask_sentence[0] = 1.
            global_mask.append(global_mask_sentence)
            index_HTMLelement = sum([tokens.shape[0] for tokens in input_text])
            index_HTMLelements.append(index_HTMLelement)
            has_answer = sentence['has_answer'] != -1
            if yesno == True:
                if sentence['string'] in yes_evidences:
                    entails_yes = True
                else:
                    entails_yes = False
                if sentence['string'] in no_evidences:
                    entails_no = True
                else:
                    entails_no = False
            else:
                entails_yes = False
                entails_no = False
            label_HTMLelements.append([int(k) * 2 - 1 for k in (int(has_answer), entails_yes, entails_no)])
            input_text.append(tokens)
            if sentence['has_answer'] != -1:
                label_answer_span.append([index_HTMLelement + sentence['answer_start'], index_HTMLelement + sentence['answer_end']])
            else:
                label_answer_span.append([-1, -1])

            mask_label_condition.append(torch.zeros(1, 5, 2))
            if sentence['condition_of'] != -1:
                mask_label_condition[-1][0, sentence['condition_of'], 1] = 1
            if sentence['has_answer'] != -1:
                mask_label_condition[-1][0, sentence['has_answer'], 0] = 1
            mask_label_condition.append(torch.zeros(len(tokens) - 1, 5, 2))
        
        input_text.append(input_tail)
        global_mask.append(torch.ones(input_tail.shape[0]))
        input_text= torch.concat(input_text, dim = 0)
        global_mask = torch.concat(global_mask, dim = 0)
        mask_HTMLelements = torch.zeros(4000)
        for i in index_HTMLelements:
            mask_HTMLelements[i] = 1
        mask_label_HTMLelements = torch.zeros(4000, 3)
        label_HTMLelements = torch.tensor(label_HTMLelements, dtype = torch.float)
        mask_label_HTMLelements[mask_HTMLelements == 1] = label_HTMLelements
        
        index_HTMLelements = torch.tensor(index_HTMLelements, dtype = torch.long)
        start_answer_span = index_HTMLelements.reshape(-1, 1)
        end_answer_span = torch.concat([index_HTMLelements[1:], torch.tensor([input_text.shape[0] - input_tail.shape[0]])]).reshape(-1, 1) - 1
        range_answer_span = torch.concat([start_answer_span, end_answer_span], dim=1)
        range_answer_span = range_answer_span[label_HTMLelements[:, 0] == 1]
        label_answer_span = torch.tensor(label_answer_span)
        label_answer_span = label_answer_span[label_HTMLelements[:, 0] == 1]

        mask_answer_span = torch.zeros(4000, 2)
        for i in range_answer_span:
            mask_answer_span[i[0] : i[1] + 1, :] = -1
        for i in label_answer_span:
            mask_answer_span[i[0], 0] = 1
            mask_answer_span[i[1], 1] = 1
        

        text_length = input_text.shape[0]
        input_text = torch.concat((input_text, torch.ones(4000 - text_length, dtype = torch.long)))
        global_mask = torch.concat((global_mask, torch.zeros(4000 - text_length, dtype = torch.long)))
        attn_mask = torch.concat((torch.ones(text_length), torch.zeros(4000 - text_length)))
        qa_id = torch.tensor(int(qa['id'].split('-')[1]), dtype = torch.long)
        mask_label_condition = torch.concat(mask_label_condition, dim=0)
        tmp = torch.zeros((4000, 5, 2), dtype=torch.long)
        tmp[:len(mask_label_condition)] = mask_label_condition
        mask_label_condition = tmp

        inputs.append([input_text, global_mask, attn_mask, mask_HTMLelements, mask_label_HTMLelements, mask_answer_span, qa_id, mask_label_condition])
    return inputs

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

    

        
    