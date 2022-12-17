import argparse
# import torch
# train_data = torch.load('../condqa_files/data/dev_inputs')
# A = [i[2].count_nonzero() for i in train_data]
# A = torch.tensor(A)
from copy import copy
from tqdm import tqdm
import torch
import json
import re
import os
from utils import Tokenizer
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str)
parser.add_argument('--model_root', type=str)
parser.add_argument('--max_len', type=int, default=1500)
parser.add_argument('--easy_passage', action='store_true')
parser.add_argument('--yesno_only', action='store_true')
parser.add_argument('--conditional_only', action='store_true')

args = parser.parse_args()

def check_same(token_id_1,token_id_2):
    if tokenizer.decode(token_id_1).lower() == tokenizer.decode(token_id_2).lower():
        return True
    return False

def get_answer_position(sentence, answer):
    answer = tokenizer(answer)
    sentence = tokenizer(sentence)
    for i in range(len(sentence) - len(answer) + 1):
        flag = True
        for j in range(len(answer)):
            if not check_same(sentence[i+j], answer[j]):
                flag = False
                break
        if flag == True:
            return i, i + len(answer) - 1
    return -1

def prepare_data(args):
    documents = json.load(open(os.path.join(args.data_root, 'documents.json')))
    documents = {i['url']:i for i in documents}
    train_data = json.load(open(os.path.join(args.data_root, 'train.json')))
    dev_data = json.load(open(os.path.join(args.data_root, 'dev.json')))
    return documents, train_data, dev_data


def encode_documents(documents):
    for k, v in tqdm(documents.items(), total = len(documents), desc = 'tokenizing documents..'):
        title = [v['title'], tokenizer(v['title'])]
        contents = [[sentence, tokenizer(sentence)] for sentence in v['contents']]
        documents.update({k: {'title': title, 'contents': contents}})
    
def convert_data_to_examples(documents: dict, data: list[dict], tokenizer) -> list[dict]: # 注释doc里的每句话,在data里面以document的形式呈现    

    examples = []
    # example format: {'title': [string, tokens], 
    #           'document': list[[string, tokens, label, is_evidence, has_answer, answer_start, answer_end, condition_of]]
    #   }
    for qa in tqdm(data, total = len(data), desc = 'annotating data..'):
        example = {}
        url = qa['url']
        example.update({'title': documents[url]['title']})
        example.update({'question': qa['question']})
        example.update({'scenario': qa['scenario']})
        example.update({'answers': qa['answers']})
        example.update({'evidences': qa['evidences']})
        example.update({'id': qa['id']})
        example.update({'document': []})

        for sent_index, (sentence, tokens) in enumerate(documents[url]['contents']):
            # iterate over document sentences to annotate all of them.
            anno = {'string': sentence, 'tokens': tokens, 'label':re.findall('<.*?>', sentence)[0],
                'is_evidence':'irrelevant', 'has_answer':-1, 'answer_start':-1, 'answer_end':-1, 'condition_of':-1}

            #annotating evidences, answers and conditions
            if sentence in qa['evidences']:
                # annotate evidences. we have 'entailed', 'contains' and 'irrelevant'. 
                # their meanings are 'itself is evidence', 'the descendants have at least an evidence', 
                # 'neither itself nor its descendants have evidences'.
                anno.update({'is_evidence':'entailed'})
                for ans_index, (answer, conditions) in enumerate(qa['answers']):
                    # annotate conditions. we can only annotate it to be the last answer's condition if there are multiple answers
                    if sentence in conditions:
                        anno.update({'condition_of':ans_index})
                    # annotate answers. we can only annotate it to be the last one if there are multiple answers
                    if answer.lower() in sentence.lower() and (answer not in ['yes', 'no']):
                        anno['has_answer'] = ans_index
                        position = get_answer_position(sentence, answer)
                        if position != -1: # normal
                            anno.update({'answer_start': position[0], 'answer_end': position[1]})

                        else: # add space before answer
                            position = get_answer_position(sentence, ' ' + answer)
                            if position != -1: # succeed
                                anno.update({'answer_start': position[0], 'answer_end': position[1]})
                            else: # add space before and after answer, and retokenize the sentence consequently
                                sentence_shift = (
                                    sentence[:sentence.lower().index(answer.lower())]
                                      + ' ' + answer + ' ' + 
                                      sentence[sentence.lower().index(answer.lower()) + len(answer):]
                                    )
                                position = get_answer_position(sentence_shift, ' ' + answer)
                                anno.update({'tokens': tokenizer(sentence_shift), 
                                    'answer_start': position[0], 'answer_end': position[1]})
            example['document'].append(anno)

        # relabel the evidences's parent nodes to be "contains"
        for sent_index, anno in enumerate(example['document']):
            if anno['is_evidence'] == 'entailed':
                status_check = {'<h1>': False, '<h2>': False, '<h3>': False, '<h4>': False, '<p>': False, '<li>': False, '<tr>': False}
                hierarchy_number = {'<h1>': 1, '<h2>': 2, '<h3>': 3, '<h4>': 4, '<p>': 5, '<li>': 6, '<tr>': 6}
                for label in status_check:
                    if hierarchy_number[label] >= hierarchy_number[anno['label']]:
                        status_check[label] = True
                #iterate back until the h1 sentence that contains the evidence
                for anno_check in example['document'][sent_index - 1: : -1]:
                    if status_check[anno_check['label']] == False and status_check['<h1>'] == False:
                        status_check[anno_check['label']] = True
                        anno_check['is_evidence'] = 'contains'
                    if status_check['<h1>'] == True:
                        break
        examples.append(example)
    return examples

def chopup_long_example(qa, max_len):
    overlap = 5
    return_qas = []
    subdoc_index = [0]
    current_len = 0
    for idx in range(len(qa['document'])):
        current_len += qa['document'][idx]['tokens'].shape[0]
        if current_len > max_len:
            subdoc_index.append(idx + 1)
            current_len = 0
    subdoc_index.append(len(qa['document']))

    for start_index, end_index in zip(subdoc_index[:-1], subdoc_index[1:]):
        return_qa = copy(qa)
        return_qa['document'] = qa['document'][max(start_index - overlap, 0): end_index]
        return_qas.append(return_qa)
    return return_qas

def convert_examples_to_inputs(examples, tokenizer, args):
    examples_short = []
    for qa in tqdm(examples, desc = 'checking long example..'):
        examples_short += chopup_long_example(qa, args.max_len)
    inputs = []

    for qa in tqdm(examples_short, desc = 'converting examples to inputs..'):
        answers = [i[0] for i in qa['answers']]
        if args.conditional_only:
            if not any(i[1] != [] for i in qa['answers']):
                continue
        if args.yesno_only:
            if ('yes' not in answers) and ('no' not in answers):
                continue
        input_head = tokenizer('<s><yes><no>')
        input_tail = tokenizer('Title: ' + qa['title'][0] + ' Question: ' + qa['question'] + ' Scenario: ' + qa['scenario']) 

        input_ids = [input_head]
        global_mask = [torch.ones(input_head.shape[0])]
        index_heads = [1, 2]

        ### initialize for yes/no answers
        label_evidence = [-1, -1]
        label_answers = [-1, -1]
        label_answer_span = []
        label_condition = [torch.zeros(5, 2), torch.zeros(5, 2)]

        answers = [i[0] for i in qa['answers']]
        if 'yes' in answers:
            label_condition[0][answers.index('yes')][0] = 1
            label_answers[0] = 1
        if 'no' in answers:
            label_condition[1][answers.index('no')][0] = 1
            label_answers[1] = 1
        

        # 占位置
        tokens = tokenizer('<h1></h1>')
        global_mask_sentence = torch.zeros(tokens.shape[0])
        global_mask_sentence[0] = 1.
        global_mask.append(global_mask_sentence)
        index_head = sum([token.shape[0] for token in input_ids])
        index_heads.append(index_head)
        label_evidence.append(-1)
        label_answers.append(-1)        
        label_condition.append(torch.zeros(5, 2))
        input_ids.append(tokens)
        
        ### iterate in article sentences
        for sentence in qa['document']:
            if args.easy_passage:
                if sentence['is_evidence'] == 'irrelevant':
                    continue
            tokens = sentence['tokens']
            global_mask_sentence = torch.zeros(tokens.shape[0])
            global_mask_sentence[0] = 1.
            global_mask.append(global_mask_sentence)
            index_head = sum([token.shape[0] for token in input_ids])
            index_heads.append(index_head)

            if sentence['string'] in qa['evidences']:
                label_evidence.append(1)
            else:
                label_evidence.append(-1)

            if sentence['has_answer'] != -1:
                label_answers.append(1)
                label_answer_span.append([index_head + sentence['answer_start'], index_head + sentence['answer_end']])
            else:
                label_answers.append(-1)
            
            label_condition.append(torch.zeros(5, 2))
            if sentence['condition_of'] != -1:
                label_condition[-1][sentence['condition_of'], 1] = 1
            if sentence['has_answer'] != -1:
                label_condition[-1][sentence['has_answer'], 0] = 1

            input_ids.append(tokens)

        input_ids.append(input_tail)
        global_mask.append(torch.ones(input_tail.shape[0]))
        input_ids = torch.concat(input_ids)
        global_mask = torch.concat(global_mask)
        ### form padded masks for four kinds of labels
        def to_tensor(list):
            if len(list) > 0 and type(list[0]) == torch.Tensor:
                return torch.stack(list)
            else:
                return torch.tensor(list)
        label_evidence = to_tensor(label_evidence)
        label_answers = to_tensor(label_answers)
        label_answer_span = to_tensor(label_answer_span)
        label_condition = to_tensor(label_condition)
        
        mask_heads = torch.zeros(4000)
        for i in index_heads:
            mask_heads[i] = 1
        
        mask_label_evidence = torch.zeros(4000, dtype=torch.long)
        mask_label_answers = torch.zeros(4000, dtype=torch.long)
        mask_label_condition = torch.zeros(4000, 5, 2)

        mask_label_evidence[mask_heads == 1] = label_evidence
        mask_label_answers[mask_heads == 1] = label_answers
        mask_label_condition[mask_heads == 1] = label_condition
        
        # define ans_range_span for span prediction
        index_heads = torch.tensor(index_heads, dtype = torch.long)
        start_answer_span = index_heads
        end_answer_span = torch.concat([index_heads[1:], torch.tensor([input_ids.shape[0] - input_tail.shape[0]])]) - 1
        range_answer_span = torch.stack([start_answer_span, end_answer_span], dim=1)
        range_answer_span = range_answer_span[label_answers == 1]

        mask_label_answer_span = torch.zeros(4000, 2)
        for i in range_answer_span:
            mask_label_answer_span[i[0] : i[1] + 1] = -1
        for i in label_answer_span:
            mask_label_answer_span[i[0], 0] = 1
            mask_label_answer_span[i[1], 1] = 1
        
        ### form padded tensors for inputs
        def pad(tensor, padding_id):
            if padding_id == 1:
                return torch.concat([tensor, torch.ones(4000 - tensor.shape[0], dtype=tensor.dtype)])
            elif padding_id == 0:
                return torch.concat([tensor, torch.zeros(4000 - tensor.shape[0], dtype=tensor.dtype)])

        attn_mask = pad(torch.ones_like(input_ids), 0)
        global_mask = pad(global_mask, 0)
        input_ids = pad(input_ids, 1)


        if any([i[1] != [] for i in qa['answers']]):
            conditional_bool = torch.ones(1)
        else:
            conditional_bool = torch.zeros(1)
        qa_id = torch.tensor(int(qa['id'].split('-')[1]), dtype = torch.long)


        inputs.append([input_ids.long(), global_mask.bool(), attn_mask.bool(), mask_heads.bool(), # inputs
            mask_label_evidence.short(), mask_label_answers.short(), mask_label_answer_span.short(), mask_label_condition.bool(), #labels 
            conditional_bool, qa_id])
    return inputs

if __name__ == '__main__':
    tokenizer = Tokenizer(args.model_root)
    documents, train_data, dev_data = prepare_data(args)
    if os.path.exists(os.path.join(args.data_root, 'tokenized_documents_and_examples')):
        print('found cached tokenized examples. loading...')
        documents, train_examples, dev_examples = torch.load(os.path.join(args.data_root, 'tokenized_documents_and_examples')).values()
    else:
        print('found no cached tokenized examples. tokenizing documents and examples...')
        encode_documents(documents)
        train_examples = convert_data_to_examples(documents, train_data, tokenizer)
        dev_examples = convert_data_to_examples(documents, dev_data, tokenizer)
        torch.save({
            'documents': documents,
            'train_examples': train_examples,
            'dev_examples': dev_examples
        }, os.path.join(args.data_root, 'tokenized_documents_and_examples'))

    train_inputs = convert_examples_to_inputs(train_examples, tokenizer, args)
    dev_inputs = convert_examples_to_inputs(dev_examples, tokenizer, args)
    torch.save(train_inputs,  os.path.join(args.data_root, 'train_inputs'))
    torch.save(dev_inputs,  os.path.join(args.data_root, 'dev_inputs'))
