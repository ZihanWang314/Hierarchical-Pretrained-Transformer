
import torch
from transformers import LongformerModel, LongformerConfig, LongformerTokenizerFast
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
from utils import Logger, input_to_batch, Tokenizer, ReIndexer, to_numpy
from contrastive_learning import ContrastiveSampler
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', default='defaultlog.txt', type=str)
parser.add_argument('--accumulation_step', default=4, type=int)
parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
parser.add_argument('--mode', type=str)
parser.add_argument('--epoch', type=int)
parser.add_argument('--data_root', type=str)
parser.add_argument('--model_root', type=str)

#model argument
parser.add_argument('--train_condition', action='store_true')
parser.add_argument('--contrastive_learning', action='store_true')
parser.add_argument('--warmup_epoch_num', type=int, default=10)
parser.add_argument('--total_epoch_num', type=int, default=100)
parser.add_argument('--contrastive_mode', type=str, default='hpt')
parser.add_argument('--nohup', action='store_true')


args = parser.parse_args()

assert args.contrastive_mode in ['hpt', 'simcse'], 'contrastive mode definition assertion'
print(f'using {args.contrastive_mode} for contrastive learning')

logger = Logger(args.logdir)

class ConditionCalculator(nn.Module):
    def __init__(self):
        super().__init__()
        self.projector = nn.Linear(768, 768)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        hidden_states_t = self.projector(hidden_states)
        hidden_states_t = self.activation(hidden_states_t)
        scores = torch.einsum('bqh,bch->bcq', hidden_states, hidden_states_t)
        scores = torch.sigmoid(scores)
        return scores

class HPTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.transformer_config = LongformerConfig.from_pretrained('allenai/longformer-base-4096')
        # self.transformer = LongformerModel(self.transformer_config)
        self.transformer = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        self.transformer.resize_token_embeddings(50282)
        self.ans_weight = 0.6
        self.span_weight = 0.2
        self.condition_weight = 0.0
        self.contrastive_weight = 0.2
        self.ans_probe = nn.Sequential(nn.Linear(768, 3), nn.Sigmoid())
        self.span_probe = nn.Sequential(nn.Linear(768, 2), nn.Sigmoid())
        self.condition_calculator = ConditionCalculator()
        
        if config.contrastive_learning:
            self.con_sampler = ContrastiveSampler(config)

    def weighted_loss(self, input, target):
        """
        return the weighted loss considering the pos-neg distribution in target data
        """
        if target.shape[0] == 0:
            return torch.tensor(0.)
        x = target.shape[0]/target.count_nonzero()/2
        y = target.shape[0]/(target.shape[0]-target.count_nonzero())/2
        tensor = torch.where(target == 1, x, y)
        loss_fn = torch.nn.BCELoss(weight=tensor)
        return loss_fn(input, target)

    def contrastive_loss(self, last_hiddens, contrastive_pairs):
        loss = []
        for hidden, pair in zip(last_hiddens, contrastive_pairs):
            pair = pair.transpose(0, 1)
            index = pair.unsqueeze(2).repeat(1, 1, hidden.shape[-1])
            hiddens_selected = torch.gather(hidden, 1, index)
            origin_hidden, new_hidden = hiddens_selected[0], hiddens_selected[1]
            hidden_similarity_map = torch.einsum('ac,bc->ab', new_hidden, origin_hidden) / (768 ** 0.5)
            denominator = hidden_similarity_map.exp().sum(1)
            contrastive_similarity = (origin_hidden * new_hidden).sum(-1) / (768 ** 0.5)
            numerator = contrastive_similarity.exp()
            pointwise_loss = - torch.log (numerator / (denominator + 1e-8))
            contrastive_loss = pointwise_loss.mean()
            loss.append(contrastive_loss)
        loss = sum(loss) / len(loss)
        return loss


    def forward(self, data, autocast = True):
        if self.config.contrastive_learning:
            data, contrastive_pairs = self.con_sampler.batch_generate(data)
            # contrastive pairs is a list: [torch.Tensor, ...] where each tensor is like: torch.tensor([[1, 2], [37, 121], [133, 66]]) \
            # where each pair is the transformed HTMLElement index from A to B 
            # the enhanced tensors be like: [origin_1, enhanced_1, origin_2, enhanced_2]
            # the enhanced data only participate in contrastive learning part, not QA part

        input_ids = data[0] # [[101, 1, ..]]
        global_masks = data[1].float() # [[1, 0, 1, 0, 1, 1, 1]]
        attn_masks = data[2].float() # [[1, 1, 1, 0, 1, 1, 1]]
        mask_HTMLelements = data[3].float() # [[..., 1, 0, 1, 0, 0]]
        mask_label_HTMLelements = data[4] # [[..., -1, 0, 1, 0, 0]]
        mask_answer_span = data[5] # [[0, 0, 1, 0], [0, 0, 0, 1]]
        mask_label_condition = data[7].float()
        attn_masks, input_ids, global_masks, mask_HTMLelements, mask_label_HTMLelements, mask_answer_span = \
            dynamic_padding(attn_masks, input_ids, global_masks, mask_HTMLelements, mask_label_HTMLelements, mask_answer_span)
        if autocast == True:
            with torch.cuda.amp.autocast():
                last_hidden_all = self.transformer(input_ids, global_attention_mask = global_masks, attention_mask = attn_masks).last_hidden_state
        else:
            last_hidden_all = self.transformer(input_ids, global_attention_mask = global_masks, attention_mask = attn_masks).last_hidden_state
        if self.config.contrastive_learning:
            last_hidden = last_hidden_all[::2]
            mask_HTMLelements = mask_HTMLelements[::2]
            mask_label_HTMLelements = mask_label_HTMLelements[::2]
            mask_answer_span = mask_answer_span[::2]
            mask_label_condition = mask_label_condition[::2]
        else:
            last_hidden = last_hidden_all

        label_extractive = mask_label_HTMLelements
        pred_extractive = self.ans_probe(last_hidden)[label_extractive != 0]
        label_extractive = (label_extractive[label_extractive != 0] + 1) / 2
        loss_extractive = self.weighted_loss(pred_extractive, label_extractive)
        
        pred_span = self.span_probe(last_hidden)[mask_answer_span != 0]
        label_span = (mask_answer_span[mask_answer_span != 0] + 1) / 2
        loss_span = self.weighted_loss(pred_span, label_span)
        indexer = ReIndexer()
        indexer.set_index(mask_HTMLelements)
        max_HTML_num = indexer.mask_r.count_nonzero(-1).max()
        mask_r = indexer.mask_r[:, :max_HTML_num]
        last_hidden_r = indexer.re_index(last_hidden)[:, :max_HTML_num] # B * HTML * H
        pred_condition = self.condition_calculator(last_hidden_r)
        #B * HTML * ANS_NUM * (ANS? COND?)
        ans_indicator = mask_label_condition[..., 0]
        cond_indicator = mask_label_condition[..., 1]
        ans_indicator = indexer.re_index(ans_indicator)[:, :max_HTML_num].transpose(-2, -1)
        cond_indicator = indexer.re_index(cond_indicator)[:, :max_HTML_num].transpose(-2, -1)
        cond_indicator = cond_indicator.unsqueeze(2).repeat(1, 1, max_HTML_num, 1)# 1, 2, 3, 3
        label_condition = torch.einsum('abcd,abc->acd',cond_indicator, ans_indicator)
        mask_r = torch.einsum('ab,ac->abc', mask_r, mask_r)
        pred_condition = pred_condition[mask_r!=0]
        label_condition = label_condition[mask_r!=0]
        loss_condition = self.weighted_loss(pred_condition, label_condition)
        
        loss_total = self.ans_weight * loss_extractive + self.span_weight * loss_span + self.condition_weight * loss_condition
        if self.config.contrastive_learning:
            last_hidden_resized = last_hidden_all.reshape(last_hidden_all.shape[0] // 2, 2, last_hidden_all.shape[1], last_hidden_all.shape[2])
            contrastive_loss = self.contrastive_loss(last_hidden_resized, contrastive_pairs)
            loss_total += self.contrastive_weight + contrastive_loss
        return loss_total, to_numpy(loss_extractive, loss_span, contrastive_loss)


class HPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = HPTEncoder(config)
        self.mode = None
        self.lr_warmup_weight = min((config.epoch + 1) / config.warmup_epoch_num, 1) # first 10 epoches use warmup
        self.lr_decay_weight = min(1, (config.total_epoch_num - config.epoch) / (config.total_epoch_num - config.warmup_epoch_num))
        # self.lr_warmup_weight = 1
        self.optimizer = torch.optim.AdamW([
            {'params':self.encoder.parameters(), 'lr':3e-5 * self.lr_warmup_weight * self.lr_decay_weight, 'weight_decay':0.01},
        ])
        self.optimizer.zero_grad()
        self.scaler = torch.cuda.amp.GradScaler()
            
    def activate_training_mode(self):
        # train with DDP
        if self.mode != None:
            self.activate_normal_mode()
        self.to('cpu')
        self.cuda(args.local_rank)
        self.encoder = torch.nn.parallel.DistributedDataParallel(
            self.encoder, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
            )
        self.encoder.module.transformer.train()
        self.mode = 'train'

    def activate_inference_mode(self):
        # inference with DP
        if self.mode != None:
            self.activate_normal_mode()
        self.to('cuda:0')
        self.encoder.transformer = torch.nn.DataParallel(self.encoder.transformer, device_ids=[0,1,2,3])
        self.encoder.transformer.eval()
        self.mode = 'eval'

    def activate_normal_mode(self):
        #non-paralleled mode
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
        record_losses = []
        if self.config.nohup:
            train_inputs = tqdm(train_inputs, total=len(train_inputs))
        for batch in train_inputs:
            batch_steps += 1
            loss, rec_l = self.encoder(batch, autocast = True)
            record_losses.append(rec_l)
            loss /= args.accumulation_step
            self.scaler.scale(loss).backward()
            if (batch_steps + 1) % args.accumulation_step == 0:
                self.scaler.unscale_(self.optimizer)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        self.activate_normal_mode()
        torch.save(self.state_dict(), os.path.join(args.model_root, 'model_current.pt'))


    def test(self, dev_inputs):
        '''
        develop on the dev set to get best threshold for test set
        '''
        self.activate_inference_mode()

        preds = {'extractive': [], 'yes': [], 'no': [], 'condition': []}
        labels = {'extractive': [], 'yes': [], 'no': [], 'condition': []}
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for data in dev_inputs:
                    input_ids = data[0]
                    global_masks = data[1].float()
                    attn_masks = data[2].float()
                    mask_HTMLelements = data[3].float()
                    mask_label_HTMLelements = data[4]
                    mask_answer_span = data[5]
                    mask_label_condition = data[7].float()


                    attn_masks, input_ids, global_masks, mask_HTMLelements, mask_label_HTMLelements, mask_answer_span = \
                        dynamic_padding(attn_masks, input_ids, global_masks, mask_HTMLelements, mask_label_HTMLelements, mask_answer_span)
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
                        
                    indexer = ReIndexer()
                    indexer.set_index(mask_HTMLelements)
                    max_HTML_num = indexer.mask_r.count_nonzero(-1).max()
                    mask_r = indexer.mask_r[:, :max_HTML_num].cuda()
                    last_hidden_r = indexer.re_index(last_hidden)[:, :max_HTML_num] # B * HTML * H
                    pred_condition = self.encoder.condition_calculator(last_hidden_r)
                    ans_indicator = mask_label_condition[..., 0]
                    cond_indicator = mask_label_condition[..., 1]
                    ans_indicator = indexer.re_index(ans_indicator)[:, :max_HTML_num].transpose(-2, -1)
                    cond_indicator = indexer.re_index(cond_indicator)[:, :max_HTML_num].transpose(-2, -1)
                    cond_indicator = cond_indicator.unsqueeze(2).repeat(1, 1, max_HTML_num, 1)# 1, 2, 3, 3
                    label_condition = torch.einsum('abcd,abc->acd',cond_indicator, ans_indicator)
                    mask_r = torch.einsum('ab,ac->abc', mask_r, mask_r)
                    mask_r = mask_r.unsqueeze(2).repeat(1, 1, ans_indicator.shape[1], 1)
                    mask_condition = torch.einsum('abcd,acd->adb', mask_r, ans_indicator)
                    pred_condition = pred_condition[mask_condition!=0].tolist()
                    label_condition = label_condition[mask_condition!=0].tolist()
                    labels['condition'] += label_condition
                    preds['condition'] += pred_condition
                    
        thresholds = []
        for name in ['extractive', 'yes', 'no', 'condition']:
            logger.log(name)
            _, threshold = analyze_binary_classification(labels[name], preds[name])
            thresholds.append(threshold)

        return thresholds

    def answering_questions(self, dev_inputs, test_inputs, tokenizer):
        '''
        making inference on test set
        '''
        self.activate_inference_mode()

        softmax_layer = torch.nn.Softmax(dim = -1)
        thre_ans, thre_yes, thre_no, thre_condition = self.test(dev_inputs)
        output = {}
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for data in test_inputs:
                    input_ids = data[0]
                    global_masks = data[1].float()
                    attn_masks = data[2].float()
                    mask_HTMLelements = data[3].float()
                    mask_label_HTMLelements = data[4]
                    mask_answer_span = data[5]
                    attn_masks, input_ids, global_masks, mask_HTMLelements, mask_label_HTMLelements, mask_answer_span = \
                        dynamic_padding(attn_masks, input_ids, global_masks, mask_HTMLelements, mask_label_HTMLelements, mask_answer_span)
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
                        id_example = data[6][sample_index].tolist()
                        pred_HTMLelements = last_hidden[index_HTMLelements]
                        pred_HTMLelements = self.encoder.ans_probe(pred_HTMLelements)
                        pred_answer_sentence_span = range_answer_span[pred_HTMLelements[:, 0] > thre_ans, :]
                        prob_pred_answers = pred_HTMLelements[pred_HTMLelements[:, 0] > thre_ans]

                        indexer = ReIndexer()
                        indexer.set_index(mask_HTMLelements[sample_index:sample_index+1])
                        HTML_num = indexer.mask_r.count_nonzero()
                        last_hidden_r = indexer.re_index(last_hidden.unsqueeze(0))[:, :HTML_num] # 1 * HTML * H
                        pred_condition = self.encoder.condition_calculator(last_hidden_r)[0] # HTML * HTML
                        pred_condition = (pred_condition > thre_condition)

                        for index_answer, range_ in enumerate(pred_answer_sentence_span):
                            pred_answer_span = softmax_layer(self.encoder.span_probe(last_hidden[range_[0]: range_[1] + 1, :]).T)
                            index_pred_answer_start = torch.argmax(pred_answer_span[0])
                            index_pred_answer_end = torch.argmax(pred_answer_span[1])
                            prob = prob_pred_answers[index_answer, 0]
                            pred_answer = tokenizer.decode(text_ids[index_pred_answer_start + range_[0]: index_pred_answer_end + range_[0] + 1])

                            answer_sentence_index = (indexer.index[0] == range_[0]).nonzero()[0][0]
                            pred_condition_index = pred_condition[answer_sentence_index]
                            pred_condition_start_end = range_answer_span[pred_condition_index]
                            pred_conditions = []
                            for start, end in pred_condition_start_end:
                                pred_conditions.append(tokenizer.decode(text_ids[start: end + 1]))


                            if pred_answer != '':
                                if id_example in output:
                                    if pred_answer not in [i[0][0] for i in output[id_example]]:
                                        output[id_example].append([[pred_answer, pred_conditions], prob])
                                    else:
                                        for i in output[id_example]:
                                            if i[0][0] == pred_answer:
                                                i[0][1] += pred_conditions
                                else:
                                    output.update({id_example:[[[pred_answer, pred_conditions], prob]]})


                        pred_yes = pred_HTMLelements[:, 1].max()
                        pred_no = pred_HTMLelements[:, 2].max()
                        if pred_yes > thre_yes:
                            if id_example in output and 'yes' not in [i[0][0] for i in output[id_example]]:
                                output[id_example].append([['yes', []], pred_yes])
                            else:
                                output.update({id_example:[[['yes', []], pred_yes]]})
                        if pred_no > thre_no:
                            if id_example in output and 'no' not in [i[0][0] for i in output[id_example]]:
                                output[id_example].append([['no', []], pred_no])
                            else:
                                output.update({id_example:[[['no', []], pred_no]]})


            output_real = []
            def answers_to_list(answers):
                answers.sort(key = lambda x:x[1], reverse = True)
                answers = [x[0] for x in answers[:5]]
                return answers
                
            for k, v in output.items():
                output_real.append({'id':'dev-'+str(k), 'answers':answers_to_list(v)})
            answered = set()
            for x in output_real:
                answered.add(int(x['id'].split('-')[1]))
            all_qa = set(range(0, 285))
            for k in all_qa - answered:
                output_real.append({'id':'dev-'+str(k), 'answers':[]})

            json.dump(output_real, open('output','w'))
            A = evaluate('output',os.path.join(args.data_root, 'dev.json'))
            metric_now = A['total']['EM'] + A['total']['F1']
            logger.log('metric: ' + str(A))
            logger.log('total: %.6f'%(metric_now))
            
        return metric_now




def dynamic_padding(attn_masks, *args):
    input_lengths = attn_masks.count_nonzero(dim = 1)
    length = input_lengths.max()
    return [attn_masks[:, :length]] + [i[:, :length] for i in args]


    
def analyze_binary_classification(label_answer_sentence, pred_answer_sentence):
    fpr, tpr, thresholds = metrics.roc_curve(label_answer_sentence, pred_answer_sentence, pos_label = 1)
    auc = metrics.auc(fpr, tpr)
    p, r, thresholds = metrics.precision_recall_curve(label_answer_sentence, pred_answer_sentence, pos_label = 1)
    f1_score = torch.tensor(2 *p * r / (p + r + 0.00001))
    logger.log('f1_score:%.4f, auc:%.4f, threshold for best_f1:%.4f, prec:%.4f, recall:%.4f'%\
        (f1_score.max(), auc, thresholds[f1_score.argmax()], p[f1_score.argmax()], r[f1_score.argmax()]))
    return (f1_score.max() + auc, thresholds[f1_score.argmax()])


if __name__ == '__main__':
    tokenizer = Tokenizer(args.model_root)
    # train_inputs = convert_examples_to_inputs(train_examples)
    # dev_inputs = convert_examples_to_inputs(dev_examples)
    # torch.save(train_inputs, 'data/train_inputs')
    # torch.save(dev_inputs, 'data/dev_inputs')
    start = args.epoch
    if args.mode == 'train':
        train_inputs = torch.load(os.path.join(args.data_root, 'train_inputs'))
        train_inputs = train_inputs[: len(train_inputs) // 10 * 8]
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        shuffle(train_inputs)
        train_inputs = input_to_batch(train_inputs, batch_size = 1, distributed = True)
        config = args
        model = HPTModel(config)
        if start > 0:
            model.load_state_dict(torch.load(os.path.join(args.model_root, 'model_current.pt'), map_location='cpu'))
        else:
            print('initializing model from longformer-base-4096')
        model.train(train_inputs)

    elif args.mode == 'inference':
        train_inputs = torch.load(os.path.join(args.data_root, 'train_inputs'))
        train_inputs = train_inputs[len(train_inputs) // 10 * 8: ]
        dev_inputs = torch.load(os.path.join(args.data_root, 'dev_inputs'))
        train_inputs = [[j.cuda() for j in i] for i in train_inputs]
        dev_inputs = [[j.cuda() for j in i] for i in dev_inputs] 
        train_inputs = input_to_batch(train_inputs, batch_size = 18, distributed = False)
        dev_inputs = input_to_batch(dev_inputs, batch_size = 6, distributed = False)
        config = args
        model = HPTModel(config)
        model.load_state_dict(torch.load(os.path.join(args.model_root, 'model_current.pt'), map_location='cpu'))

        logger.log(f'epoch_{start + 1}')
        metric = model.answering_questions(train_inputs, dev_inputs, tokenizer)
        try:
            with open(os.path.join(args.model_root, 'result.txt'), 'r') as file:
                best_performance = float(file.readlines()[0])
        except:
            best_performance = 0
        if metric > best_performance:
            model.activate_normal_mode()
            torch.save(model.state_dict(), os.path.join(args.model_root, 'model_best.pt'))
            logger.log('achieving best result, now saving to model_best.pt')
            with open(os.path.join(args.model_root, 'result.txt'), 'w') as file:
                file.write(str(metric))

