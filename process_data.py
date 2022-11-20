import argparse
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
parser.add_argument('--doc_length', type=int, default=150)
parser.add_argument('--doc_overlap', type=int, default=120)

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

def chopup_long_example(qa, length, step):
    return_qas = []
    subdoc_index = [0]
    while subdoc_index[-1] + length < len(qa['document']):
        subdoc_index.append(subdoc_index[-1] + step)
    for start_index in subdoc_index:
        return_qa = copy(qa)
        return_qa['document'] = qa['document'][start_index: start_index + length]
        return_qas.append(return_qa)
    return return_qas

def convert_examples_to_inputs(examples, tokenizer, args):
    examples_short = []
    for qa in tqdm(examples, desc = 'checking long example..'):
        examples_short += chopup_long_example(qa, args.doc_length, args.doc_overlap)
    inputs = []

    for qa in tqdm(examples_short, desc = 'converting examples to inputs..'):
        input_head = tokenizer('Title: ' + qa['title'][0] + ' Document: ')
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

if __name__ == '__main__':
    tokenizer = Tokenizer(args.model_root)
    documents, train_data, dev_data = prepare_data(args)
    # if os.path.exists(os.path.join(args.data_root, 'tokenized_documents_and_examples')):
    #     print('found cached tokenized examples. loading...')
    #     documents, train_examples, dev_examples = torch.load(os.path.join(args.data_root, 'tokenized_documents_and_examples')).values()
    if False:
        pass
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
