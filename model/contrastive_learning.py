import torch
from utils import Tokenizer
tokenizer = Tokenizer('../condqa_old/model')

from utils import TxtNode, get_level
import numpy
import random


def split_tokens(tokens):
    tokens = tokens.tolist()
    cutoffs = []
    current_cutoff = []
    for idx, i in enumerate(tokens):
        if i not in range(50265, 50272):
            current_cutoff.append((i, idx))
        else:
            cutoffs.append(current_cutoff)
            current_cutoff = [(i, idx)]
    cutoffs.append(current_cutoff)
    return cutoffs

def create_node_from_document(document_tokens):
    base_node = TxtNode([(0, -1)], tokenizer)
    nodes = [base_node]
    current_node = base_node
    for segment in document_tokens:
        while current_node.level >= get_level(segment):
            current_node = current_node.parent

        child = TxtNode(segment, tokenizer)
        nodes.append(child)
        current_node.children.append(child)
        child.parent = current_node
        current_node = child
    return base_node

def get_grouped_tokens(input_ids):
    splited_tokens = split_tokens(input_ids)
    start_tokens = splited_tokens[0]
    document_tokens = splited_tokens[1:-1]
    end_tokens = [i for i in splited_tokens[-1] if i[0] != 0]
    l_document_tokens = []
    while True:
        token = end_tokens.pop(0)
        if token[0] in list(range(50272, 50279)):
            l_document_tokens.append(token)
            break
        else:
            l_document_tokens.append(token)
    document_tokens.append(l_document_tokens)
    return start_tokens, document_tokens, end_tokens


def repeat_a_node(base_node):
    nodes = base_node.get_nodes_list()
    nodes = [node for node in nodes if node.parent != None]
    node = random.choice(nodes)
    node_new = node.copy()
    parent = node.parent
    node_new.parent = parent
    position = random.choice(list(range(len(parent.children) + 1)))
    if len(node_new) + len(base_node) < 3800:
        parent.children = parent.children[:position] + [node_new] + parent.children[position:]
    return base_node

def remove_a_node(base_node):
    nodes = base_node.get_nodes_list()
    nodes = [node for node in nodes if node.parent != None]
    if len(nodes) <= 1:
        return base_node
    else:
        node = random.choice(nodes)
        parent = node.parent
        parent.children.remove(node)
    return base_node

def mask_a_node(base_node):
    nodes = base_node.get_nodes_list()
    node = random.choice(nodes)
    node.text = [(50264, j) for i, j in node.text]
    return base_node

def reorder_a_node(base_node):
    nodes = base_node.get_nodes_list()
    nodes = [node for node in nodes if len(node.children) > 1]
    node = random.choice(nodes)
    random.shuffle(node.children)
    return base_node



def recover_index_from_node(node):
    nodes = node.get_nodes_list()
    text = [i.text for i in nodes][1:]
    real_text = []
    for i in text:
        real_text += i

    origin_HTMLelement_index = [i[0][1] for i in text]
    text_lengths = [len(i) for i in text]
    generated_HTMLelement_index = [0] + numpy.cumsum(text_lengths)[:-1].tolist()
    HTMLelement_index = list(zip(origin_HTMLelement_index, generated_HTMLelement_index))
    
    return real_text, HTMLelement_index


class ContrastiveSampler:
    def __init__(self, config):
        self.config = config

    def contrastive_sampling(self, input_ids):
        start_tokens, document_tokens, end_tokens = get_grouped_tokens(input_ids)
        base_node = create_node_from_document(document_tokens)
        if self.config.contrastive_mode == 'simcse':
            pass
        elif self.config.contrastive_mode == 'hpt':
            for i in range(len(base_node.get_nodes_list()) // 20):
                base_node = repeat_a_node(base_node)

            for i in range(len(base_node.get_nodes_list()) // 20):
                base_node = remove_a_node(base_node)
                if len(base_node.children) == 1: # stop removing
                    break

            for i in range(len(base_node.get_nodes_list()) // 5):
                base_node = reorder_a_node(base_node)

            for i in range(len(base_node.get_nodes_list()) // 10):
                base_node = mask_a_node(base_node)

        document_tokens, HTMLelement_index = recover_index_from_node(base_node)
        HTMLelement_index = torch.tensor(HTMLelement_index)
        document_tokens = start_tokens + document_tokens + end_tokens
        HTMLelement_index[:, 1] += len(start_tokens)

        return document_tokens, HTMLelement_index

    def generate_contrastive_sample(self, sample):
        input_ids = sample[0]
        global_mask = sample[1]
        attention_mask = sample[2]
        mask_HTMLelements = sample[3]
        mask_label_HTMLelements = sample[4]
        mask_answer_span = sample[5]
        qa_id = sample[6]
        mask_label_condition = sample[7]

        new_input, contrastive_pairs = self.contrastive_sampling(input_ids)
        new_input_ids = torch.tensor([i[0] for i in new_input])
        arrangement_index = torch.tensor([i[1] for i in new_input])

        global_mask = global_mask[arrangement_index]
        attention_mask = attention_mask[arrangement_index]
        mask_HTMLelements = mask_HTMLelements[arrangement_index]
        mask_label_HTMLelements = mask_label_HTMLelements[arrangement_index]
        mask_answer_span = mask_answer_span[arrangement_index]
        mask_label_condition = mask_label_condition[arrangement_index]

        text_length = new_input_ids.shape[0]
        new_input_ids = torch.concat((new_input_ids, torch.zeros(4000 - text_length, dtype = torch.long)))
        global_mask = torch.concat((global_mask, torch.zeros(4000 - text_length, dtype = torch.bool)))
        attention_mask = torch.concat((attention_mask, torch.zeros(4000 - text_length, dtype = torch.bool)))
        mask_HTMLelements = torch.concat((mask_HTMLelements, torch.zeros(4000 - text_length, dtype = torch.bool)))
        mask_label_HTMLelements = torch.concat((mask_label_HTMLelements, torch.zeros((4000 - text_length, 3), dtype = torch.long)))
        mask_answer_span = torch.concat((mask_answer_span, torch.zeros((4000 - text_length, 2), dtype = torch.long)))
        mask_label_condition = torch.concat((mask_label_condition, torch.zeros((4000 - text_length, 5, 2), dtype = torch.bool)))

        new_sample = [new_input_ids, global_mask, attention_mask, mask_HTMLelements, mask_label_HTMLelements, \
            mask_answer_span, qa_id, mask_label_condition]
        return new_sample, contrastive_pairs


    def batch_generate(self, batch):
        input_ids = []
        global_mask = []
        attention_mask = []
        mask_HTMLelements = []
        mask_label_HTMLelements = []
        mask_answer_span = []
        qa_id = []
        mask_label_condition = []
        contrastive_pairs = []

        for i in range(len(batch[0])):
            sample = [batch[0][i].cpu(), batch[1][i].cpu(), batch[2][i].cpu(), batch[3][i].cpu(), \
                batch[4][i].cpu(), batch[5][i].cpu(), batch[6][i].cpu(), batch[7][i].cpu()]
            new_sample, contrastive_pair = self.generate_contrastive_sample(sample)
            input_ids += [sample[0], new_sample[0]]
            global_mask += [sample[1], new_sample[1]]
            attention_mask += [sample[2], new_sample[2]]
            mask_HTMLelements += [sample[3], new_sample[3]]
            mask_label_HTMLelements += [sample[4], new_sample[4]]
            mask_answer_span += [sample[5], new_sample[5]]
            qa_id += [sample[6], new_sample[6]]
            mask_label_condition += [sample[7], new_sample[7]]
            contrastive_pairs.append(contrastive_pair)
        
        device = batch[i][0].device
        input_ids = torch.stack(input_ids).to(device)
        global_mask = torch.stack(global_mask).to(device)
        attention_mask = torch.stack(attention_mask).to(device)
        mask_HTMLelements = torch.stack(mask_HTMLelements).to(device)
        mask_label_HTMLelements = torch.stack(mask_label_HTMLelements).to(device)
        mask_answer_span = torch.stack(mask_answer_span).to(device)
        qa_id = torch.stack(qa_id).to(device)
        mask_label_condition = torch.stack(mask_label_condition).to(device)
        contrastive_pairs = [i.to(device) for i in contrastive_pairs]

        batch_new = [input_ids, global_mask, attention_mask, mask_HTMLelements, mask_label_HTMLelements, mask_answer_span, qa_id, mask_label_condition]
        return batch_new, contrastive_pairs


