
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
from data.evaluate import evaluate
import argparse
from utils import Logger
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
logger = Logger('new_log_11152022.txt')

class HPTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        self.transformer.resize_token_embeddings(50282)
        self.ans_weight = 0.7
        self.span_weight = 0.3
        self.ans_probe = nn.Sequential(nn.Linear(768, 3), nn.Sigmoid())
        self.span_probe = nn.Sequential(nn.Linear(768, 2), nn.Sigmoid())

    def weighted_loss(self, input, target):
        """
        return the weighted loss considering the pos-neg distribution in target data
        """
        x = target.shape[0]/target.count_nonzero()/2
        y = target.shape[0]/(target.shape[0]-target.count_nonzero())/2
        tensor = torch.where(target == 1, x, y)
        loss_fn = torch.nn.BCELoss(weight=tensor)
        return loss_fn(input, target)

    def forward(self, data, autocast = True):
        input_ids = data[0]
        global_masks = data[1]
        attn_masks = data[2]
        mask_HTMLelements = data[3]
        mask_label_HTMLelements = data[4]
        mask_answer_span = data[5]
        attn_masks, input_ids, global_masks, mask_HTMLelements, mask_label_HTMLelements, mask_answer_span = dynamic_padding(attn_masks, input_ids, global_masks, mask_HTMLelements, mask_label_HTMLelements, mask_answer_span)
        if autocast == True:
            with torch.cuda.amp.autocast():
                last_hidden = self.transformer(input_ids, global_attention_mask = global_masks, attention_mask = attn_masks).last_hidden_state
        else:
            last_hidden = self.transformer(input_ids, global_attention_mask = global_masks, attention_mask = attn_masks).last_hidden_state
        label_extractive = mask_label_HTMLelements
        pred_extractive = self.ans_probe(last_hidden)[label_extractive != 0]
        label_extractive = (label_extractive[label_extractive != 0] + 1) / 2
        loss_extractive = self.weighted_loss(pred_extractive, label_extractive)
        
        pred_span = self.span_probe(last_hidden)[mask_answer_span != 0]
        label_span = (mask_answer_span[mask_answer_span != 0] + 1) / 2
        loss_span = self.weighted_loss(pred_span, label_span)

        return self.ans_weight * loss_extractive + self.span_weight * loss_span


class HPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = HPTEncoder(config)
        self.mode = None
        self.optimizer = torch.optim.AdamW([
            {'params':self.encoder.parameters(), 'lr':3e-5, 'weight_decay':0.01},
        ])
        self.optimizer.zero_grad()
        self.scaler = torch.cuda.amp.GradScaler()
            
    def activate_training_mode(self):
        # train with DDP
        if self.mode != None:
            self.activate_normal_mode()
        self.to('cpu')
        self.cuda(args.local_rank)
        self.encoder = torch.nn.parallel.DistributedDataParallel(self.encoder, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        self.encoder.module.transformer.train()
        self.mode = 'train'

    def activate_inference_mode(self):
        if self.mode != None:
            self.activate_normal_mode()
        self.to('cuda:0')
        self.encoder.transformer = torch.nn.DataParallel(self.encoder.transformer, device_ids=[0,1,2,3])
        self.encoder.transformer.eval()
        self.mode = 'eval'

    def activate_normal_mode(self):
        if self.mode == 'train':
            self.encoder = self.encoder.module
        elif self.mode == 'eval':
            self.encoder.transformer = self.encoder.transformer.module
        self.encoder.transformer.eval()
        self.to('cpu')
        self.mode = None

    def train(self, train_inputs):
        self.optimizer.zero_grad()
        self.activate_training_mode()
        batch_steps = 0
        for batch in tqdm(list(train_inputs), total = len(train_inputs)):
            batch_steps += 1
            loss = self.encoder(batch, autocast = True)
            loss /= 4
            self.scaler.scale(loss).backward()
            if (batch_steps + 1) % 4 == 0:
                self.scaler.unscale_(self.optimizer)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        self.activate_normal_mode()
        torch.save(self.state_dict(), 'model/model_current.pt')


    def test(self, dev_inputs, proportion = 1):
        logger.log('evaluating.')
        self.activate_inference_mode()

        preds = {'extractive': [], 'yes': [], 'no': []}
        labels = {'extractive': [], 'yes': [], 'no': []}
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for data in tqdm(dev_inputs, total = len(dev_inputs), ):
                    if np.random.rand() > proportion:
                        continue
                    input_ids = data[0]
                    global_masks = data[1]
                    attn_masks = data[2]
                    mask_HTMLelements = data[3]
                    mask_label_HTMLelements = data[4]
                    mask_answer_span = data[5]
                    attn_masks, input_ids, global_masks, mask_HTMLelements, mask_label_HTMLelements, mask_answer_span = dynamic_padding(attn_masks, input_ids, global_masks, mask_HTMLelements, mask_label_HTMLelements, mask_answer_span)
                    last_hidden = self.encoder.transformer(input_ids, global_attention_mask = global_masks, attention_mask = attn_masks).last_hidden_state

                    label = mask_label_HTMLelements[:, :, 0]
                    pred = self.encoder.ans_probe(last_hidden)[:, :, 0][label != 0].tolist()
                    label = ((label[label != 0] + 1) / 2).tolist()
                    labels['extractive'] += label
                    preds['extractive'] += pred

                    label = mask_label_HTMLelements[:, :, 1]
                    pred = self.encoder.ans_probe(last_hidden)[:, :, 1][label != 0].tolist()
                    label = ((label[label != 0] + 1) / 2).tolist()
                    labels['yes'] += label
                    preds['yes'] += pred

                    label = mask_label_HTMLelements[:, :, 2]
                    pred = self.encoder.ans_probe(last_hidden)[:, :, 2][label != 0].tolist()
                    label = ((label[label != 0] + 1) / 2).tolist()
                    labels['no'] += label
                    preds['no'] += pred
                        

        thresholds = []
        for idx, name in enumerate(['extractive', 'yes', 'no']):
            logger.log(f'analyzing {name}\n')
            metric, threshold = analyze_binary_classification(labels[name], preds[name])
            thresholds.append(threshold)

        return thresholds

    def answering_questions(self, train_inputs, dev_inputs):
        self.activate_inference_mode()

        softmax_layer = torch.nn.Softmax(dim = -1)
        thre_ans, thre_yes, thre_no = self.test(train_inputs, proportion = 1)
        output = {}
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for data in tqdm(dev_inputs, total = len(dev_inputs), ):
                    input_ids = data[0]
                    global_masks = data[1]
                    attn_masks = data[2]
                    mask_HTMLelements = data[3]
                    mask_label_HTMLelements = data[4]
                    mask_answer_span = data[5]
                    attn_masks, input_ids, global_masks, mask_HTMLelements, mask_label_HTMLelements, mask_answer_span = dynamic_padding(attn_masks, input_ids, global_masks, mask_HTMLelements, mask_label_HTMLelements, mask_answer_span)
                    last_hiddens = self.encoder.transformer(input_ids, global_attention_mask = global_masks, attention_mask = attn_masks).last_hidden_state
                    sample_count = data[0].shape[0]

                    for sample_index in range(sample_count):
                        last_hidden = last_hiddens[sample_index]
                        text_ids = input_ids[sample_index].cuda(last_hidden.device)
                        index_HTMLelements = mask_HTMLelements[sample_index].nonzero().flatten()
                        global_mask = global_masks[sample_index].nonzero().flatten()
                        index_end = global_mask[global_mask > index_HTMLelements.max()].min().unsqueeze(0)

                        range_answer_span = torch.concat([
                            index_HTMLelements.reshape(-1,1), 
                            torch.concat([index_HTMLelements[1:], index_end]).reshape(-1,1) - 1
                        ], dim = 1)
                        id_example = data[-1][sample_index].tolist()
                        pred_HTMLelements = last_hidden[index_HTMLelements]
                        pred_HTMLelements = self.encoder.ans_probe(pred_HTMLelements)
                        pred_answer_sentence_span = range_answer_span[pred_HTMLelements[:, 0] > thre_ans, :]
                        prob_pred_answers = pred_HTMLelements[pred_HTMLelements[:, 0] > thre_ans]
                        for index_answer, range_ in enumerate(pred_answer_sentence_span):
                            pred_answer_span = softmax_layer(self.encoder.span_probe(last_hidden[range_[0]: range_[1] + 1, :]).T)
                            index_pred_answer_start = torch.argmax(pred_answer_span[0])
                            index_pred_answer_end = torch.argmax(pred_answer_span[1])
                            prob = prob_pred_answers[index_answer, 0]
                            pred_answer = tokenizer.decode(text_ids[index_pred_answer_start + range_[0]: index_pred_answer_end + range_[0] + 1])
                            if pred_answer != '':
                                if id_example in output:
                                    output[id_example].append([pred_answer, prob])
                                else:
                                    output.update({id_example:[[pred_answer, prob]]})

                        pred_yes = pred_HTMLelements[:, 1].max()
                        pred_no = pred_HTMLelements[:, 2].max()
                        if pred_yes > thre_yes:
                            if id_example in output:
                                output[id_example].append(['yes', pred_yes])
                            else:
                                output.update({id_example:[['yes', pred_yes]]})
                        if pred_no > thre_no:
                            if id_example in output:
                                output[id_example].append(['no', pred_no])
                            else:
                                output.update({id_example:[['no', pred_no]]})


            output_real = []
            def answers_to_list(answers):
                answers.sort(key = lambda x:x[1], reverse = True)
                answers = {x[0] for x in answers[:5]}
                return [[i,[]] for i in answers]
                
            for i in output:
                output_real.append({'id':'dev-'+str(i), 'answers':answers_to_list(output[i])})
            answered = set()
            for x in output_real:
                answered.add(int(x['id'].split('-')[1]))
            all_qa = set(range(1,285))
            for k in all_qa - answered:
                output_real.append({'id':'dev-'+str(k), 'answers':[]})

            json.dump(output_real,open('output','w'))
            A = evaluate('output','data/dev.json')
            metric_now = A['total']['EM'] + A['total']['F1']
            logger.log('metric: ' + str(A))
            logger.log('total: %.4f\n'%(metric_now))

        return metric_now


def prepare_tokenizer():
    tokenizer = LongformerTokenizerFast.from_pretrained('model/LongformerTokenizer')
    tokenizer.special_tokens_map.update({'yes_token':'<yes>','no_token':'<no>','na_token':'<na>'})
    tokenizer.add_tokens(
        ['<s>','<h1>','<h2>','<h3>','<h4>','<p>','<li>','<tr>',
        '</s>','</h1>','</h2>','</h3>','</h4>','</p>','</li>','</tr>',]
    )
    tokenizer.add_tokens(
        ['<yes>','<no>','<na>']
    )
    tokenizer.elem_tokens = tokenizer('<h1> <h2> <h3> <h4> <p> <li> <tr>').input_ids[1:-1]
    return tokenizer

def tokenize(input_str):
    return tokenizer(input_str, return_tensors = 'pt', padding='longest').input_ids[0,1:-1]

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

def convert_examples_to_inputs(examples):
    examples_short = []
    for qa in tqdm(examples, desc = 'checking long example..'):
        examples_short += check_and_transform_long_example(qa)

    inputs = []
    for qa_index, qa in tqdm(enumerate(examples_short), total = len(examples_short), desc = 'converting examples to inputs..'):
        input_head = tokenize('Title: ' + qa['title'] + ' Document: ')
        input_tail = tokenize(' Question: ' + qa['question'] + ' Scenario: ' + qa['scenario']) 

        input_text = [input_head]
        global_mask = [torch.ones(input_head.shape[0])]
        index_HTMLelements = []
        label_HTMLelements = []
        label_answer_span = []

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
        input_text = torch.concat((input_text, torch.zeros(4000 - text_length, dtype = torch.long)))
        global_mask = torch.concat((global_mask, torch.zeros(4000 - text_length, dtype = torch.long)))
        attn_mask = torch.concat((torch.ones(text_length), torch.zeros(4000 - text_length)))
        qa_id = torch.tensor(int(qa['id'].split('-')[1]), dtype = torch.long)
        inputs.append([input_text, global_mask, attn_mask, mask_HTMLelements, mask_label_HTMLelements, mask_answer_span, qa_id])
    return inputs

def dynamic_padding(attn_masks, *args):
    input_lengths = attn_masks.count_nonzero(dim = 1)
    length = input_lengths.max()
    return [attn_masks[:, :length]] + [i[:, :length] for i in args]

def input_to_batch(inputs, batch_size = 4, distributed = False):
    if distributed == False:
        batches = torch.utils.data.DataLoader(inputs, batch_size = batch_size, shuffle = True)
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(inputs)
        batches = torch.utils.data.DataLoader(inputs, batch_size=batch_size, shuffle = (sampler is None), sampler=sampler, pin_memory=True)
    return batches
    
def analyze_binary_classification(label_answer_sentence, pred_answer_sentence):
    fpr, tpr, thresholds = metrics.roc_curve(label_answer_sentence, pred_answer_sentence, pos_label = 1)
    auc = metrics.auc(fpr, tpr)
    p, r, thresholds = metrics.precision_recall_curve(label_answer_sentence, pred_answer_sentence, pos_label = 1)
    f1_score = torch.tensor(2 *p * r / (p + r + 0.00001))
    logger.log('f1_score:%.4f, auc:%.4f, threshold for best_f1:%.4f, prec:%.4f, recall:%.4f\n'%\
        (f1_score.max(), auc, thresholds[f1_score.argmax()], p[f1_score.argmax()], r[f1_score.argmax()]))
    return (f1_score.max() + auc, thresholds[f1_score.argmax()])


if __name__ == '__main__':
    tokenizer = prepare_tokenizer()

    # train_inputs = convert_examples_to_inputs(train_examples)
    # dev_inputs = convert_examples_to_inputs(dev_examples)
    # torch.save(train_inputs, 'data/train_inputs')
    # torch.save(dev_inputs, 'data/dev_inputs')
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--epoch', type=int)
    args = parser.parse_args()
    status = args.mode
    start = args.epoch

    if status == 'train':
        train_inputs = torch.load('data/train_inputs')
        train_inputs = train_inputs[: len(train_inputs) // 10 * 8]
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        shuffle(train_inputs)
        train_inputs = input_to_batch(train_inputs, batch_size = 1, distributed = True)
        config = {}
        model = HPTModel(config)
        if start > 0:
            model.load_state_dict(torch.load('model/model_current.pt', map_location='cpu'))
        logger.log('training.') 
        model.train(train_inputs)

    elif status == 'inference':
        train_inputs = torch.load('data/train_inputs')
        train_inputs = train_inputs[len(train_inputs) // 10 * 8: ]
        dev_inputs = torch.load('data/dev_inputs')
        train_inputs = [[j.cuda() for j in i] for i in train_inputs]
        dev_inputs = [[j.cuda() for j in i] for i in dev_inputs] 
        train_inputs = input_to_batch(train_inputs, batch_size = 18, distributed = False)
        dev_inputs = input_to_batch(dev_inputs, batch_size = 6, distributed = False)
        config = {}
        model = HPTModel(config)
        if start > 0:
            model.load_state_dict(torch.load('model/model_current.pt', map_location='cpu'))

        logger.log(f'epoch_{start}')
        metric = model.answering_questions(train_inputs, dev_inputs)
        logger.log('metric: ' + str(metric))
        # if metric > 1.01 + start * 0.001:
        #     model.to('cpu')
        #     model.model.transformer = model.model.transformer.module
        #     self.activate_normal_mode()
        #     torch.save(self.state_dict(), 'model/model_current.pt')

        #     torch.save(model.model, 'model/model_best_%d'%(int(100*metric)))
