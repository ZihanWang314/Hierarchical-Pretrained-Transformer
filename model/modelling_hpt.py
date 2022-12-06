
import torch
# from transformers import LongformerModel, LongformerConfig
from transformers import RobertaModel, RobertaConfig
import json
from utils import Tokenizer
from tqdm import tqdm
from torch import nn
import os
import utils
from utils.evaluate import evaluate
from utils import ReIndexer, to_numpy, analyze_binary_classification, dynamic_padding, Logger
from model.contrastive_learning import ContrastiveSampler
from typing import Optional

logger2 = Logger('nohup.out')

class ConditionCalculator(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff = nn.Sequential(nn.GELU(), nn.Linear(768, 3072))
        self.q = nn.Linear(3072, 768)
        self.k = nn.Linear(3072, 768)
        

    def forward(self, hidden_states, mask_r):
        hidden_states = self.ff(hidden_states)
        q = self.q(hidden_states)
        k = self.k(hidden_states)
        scores = torch.einsum('abh,ach->abc', q, k) / (768 ** 0.5)
        scores -= (1 - mask_r.float()[..., None]) * 1000
        
        return scores


class HierarchicalTransformerAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.relationship_bias = nn.Embedding(14, config.ha_num_heads)
        self.query = nn.Linear(config.model_hidden_size, config.ha_num_heads * config.ha_hidden_size)
        self.key = nn.Linear(config.model_hidden_size, config.ha_num_heads * config.ha_hidden_size)
        self.value = nn.Linear(config.model_hidden_size, config.ha_num_heads * config.ha_hidden_size)
        self.output = nn.Linear(config.ha_num_heads * config.ha_hidden_size, config.model_hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.layernorm = nn.LayerNorm(config.model_hidden_size)


    def forward(self, hidden_states, level_hierarchy):
        indexer = ReIndexer()
        indexer.set_index(level_hierarchy != 0)
        max_head_num = indexer.get_max_head_num()
        mask_r = indexer.get_mask_r().to(hidden_states.device)
        hidden_states_r = indexer.re_index(hidden_states)[:, :max_head_num]
        batch_size, max_num_heads, _ = hidden_states_r.shape
        hidden_states_nonglobal = indexer.re_index(hidden_states)[:, max_head_num:]

        level_hierarchy_r = indexer.re_index(level_hierarchy)[:, :max_head_num]
        relationship_unmasked = level_hierarchy_r.unsqueeze(1) - level_hierarchy_r.unsqueeze(2)
        mask = torch.einsum('ab,ac->abc', mask_r, mask_r).float()
        relationship = relationship_unmasked - (1 - mask.long()) * 10000
        relationship += 6
        relationship = torch.where(relationship > 0, relationship, 0)
        relationship_bias = self.relationship_bias(relationship.flatten()).reshape(
            batch_size, max_num_heads, max_num_heads, self.config.ha_num_heads
        )

        query = self.query(hidden_states_r).reshape(batch_size, max_num_heads, self.config.ha_num_heads, self.config.ha_hidden_size)
        key = self.key(hidden_states_r).reshape(batch_size, max_num_heads, self.config.ha_num_heads, self.config.ha_hidden_size)
        value = self.value(hidden_states_r).reshape(batch_size, max_num_heads, self.config.ha_num_heads, self.config.ha_hidden_size)
        similarity = torch.einsum('apcd,aqcd->apqc', query, key) / (self.config.ha_hidden_size ** 0.5)
        similarity -= (1 - mask.unsqueeze(-1)) * 10000
        similarity += relationship_bias
        scores = similarity.softmax(2)
        scores = self.dropout(scores)
        # scores: batch_size * target_len * source_len * num_heads
        # value: batch_size * source_len * num_heads * hidden_states
        # expect: batch_size * target_len * num_heads * hidden_states
        attention_result = torch.einsum('btsn,bsnh->btnh', scores, value).reshape(
            batch_size, max_num_heads, self.config.ha_num_heads * self.config.ha_hidden_size
            )
        attention_output = self.layernorm(self.dropout(self.output(attention_result)) + hidden_states_r)
        layer_output = torch.concat([attention_output, hidden_states_nonglobal], dim=1)
        layer_output = indexer.recover_index(layer_output)
        return layer_output




class HierarchicalTransformerEncoder(nn.Module):
    def __init__(self, config, longformer_encoder):
        super().__init__()
        self.layer = longformer_encoder.layer
        # self.head_attentions = nn.ModuleList([HierarchicalTransformerAttention(config) for _ in range(3)])

    def forward(self, hidden_states, attention_mask, padding_len, level_hierarchy):
        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        is_global_attn = is_index_global_attn.flatten().any().item()

        for idx, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                is_index_masked=is_index_masked,
                is_index_global_attn=is_index_global_attn,
                is_global_attn=is_global_attn,
            )
            hidden_states = layer_outputs[0]
            # if idx >= 9:
            #     hidden_states = self.head_attentions[idx - 9](hidden_states, level_hierarchy) # additional attention for our task

        # undo padding
        if padding_len > 0:
            hidden_states = hidden_states[:, :-padding_len]
        return hidden_states


class HierarchicalTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        modelconfig = RobertaConfig.from_pretrained('roberta-base')
        # modelconfig.attention_window = 128
        longformer = RobertaModel(modelconfig)
        checkpoint = RobertaModel.from_pretrained('roberta-base')
        longformer.load_state_dict(checkpoint.state_dict())
        longformer.resize_token_embeddings(50282)

        self._merge_to_attention_mask = longformer._merge_to_attention_mask
        self._pad_to_window_size = longformer._pad_to_window_size
        self.get_extended_attention_mask = longformer.get_extended_attention_mask

        self.embeddings = longformer.embeddings
        self.encoder = HierarchicalTransformerEncoder(config, longformer.encoder)

    def _get_hierarchy(self, input_ids):
        level_hierarchy = torch.where((0 < input_ids - 50264) & (input_ids - 50264 < 8), input_ids - 50264, 0)
        level_hierarchy[level_hierarchy == 7] = 6
        return level_hierarchy

    def forward(
        self,         
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
    ):
        device = input_ids.device
        input_shape = input_ids.size()
        attention_mask = self._merge_to_attention_mask(attention_mask, global_attention_mask)
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        padding_len, input_ids, attention_mask, token_type_ids, _, _ = self._pad_to_window_size(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=None,
            inputs_embeds=None,
            pad_token_id=1,
        )
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)[
            :, 0, 0, :
        ]
        embedding_output = self.embeddings(
            input_ids=input_ids
        )

        #define level hierarchy for the given tokens. for every given value > 0, the less the value, the higher the hierarchy
        level_hierarchy = self._get_hierarchy(input_ids)
        encoder_output = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            padding_len=padding_len,
            level_hierarchy=level_hierarchy
        )
        return encoder_output



class HPTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.evidence_weight = 0.5
        self.yesno_weight = 0
        self.extractive_weight = 0
        self.span_weight = 0
        self.condition_weight = 1
        self.contrastive_weight = 0.5 # it's ok they don't sum to be 1 because of adam
        global logger
        logger = Logger(config.logdir)
        self.transformer = HierarchicalTransformer(config)
        # self.head_attention = nn.MultiheadAttention(768, 12, 0.1, batch_first=True)
        # self.head_attention = nn.Identity() #NOTE for test

        self.evidence_probe = nn.Linear(768, 1)
        self.yes_probe = nn.Linear(768, 1)
        self.no_probe = nn.Linear(768, 1)
        self.extractive_probe = nn.Linear(768, 1)
        self.span_probe = nn.Linear(768, 2)

        self.condition_calculator = ConditionCalculator()
        
        if config.contrastive_learning:
            self.con_sampler = ContrastiveSampler(config)

    def weighted_loss(self, input, target):
        """
        return the weighted loss considering the pos-neg distribution in target data
        """
        input = input.flatten()
        target = target.flatten()
        if target.shape[0] == 0:
            return torch.tensor(torch.nan).to(input.device)
        # elif (target.count_nonzero() == 0) or (target.count_nonzero() == target.shape[0]):
        #     return torch.tensor(torch.nan).to(input.device)
        else:
            x = target.shape[0]/target.count_nonzero()/2
            y = target.shape[0]/(target.shape[0]-target.count_nonzero())/2
            tensor = torch.where(target == 1, x, y)
            loss_fn = torch.nn.BCEWithLogitsLoss(weight=tensor)
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

    def load_batch(self, data):
        input_ids = data[0].long()
        global_masks = data[1].float()
        attn_masks = data[2].float()
        mask_heads = data[3].float() # inputs
        mask_label_evidence = data[4].float()
        mask_label_answers = data[5].float()
        mask_label_answer_span = data[6].float()
        mask_label_condition = data[7].float() 
        qa_id = data[8].long()
        attn_masks, input_ids, global_masks, mask_heads, \
            mask_label_evidence, mask_label_answers, mask_label_answer_span, mask_label_condition = \
        dynamic_padding(attn_masks, input_ids, global_masks, mask_heads, \
            mask_label_evidence, mask_label_answers, mask_label_answer_span, mask_label_condition)

        return input_ids, global_masks, attn_masks, mask_heads, mask_label_evidence, mask_label_answers, \
            mask_label_answer_span, mask_label_condition, qa_id

    def answer_probe(self, hidden_states_r): # B * H * S
        y, n, e = self.yes_probe(hidden_states_r[:, 0:1]), self.no_probe(hidden_states_r[:, 1:2]), self.extractive_probe(hidden_states_r[:, 2:])
        return torch.concat([y, n, e], dim=1)

    def forward(self, data, autocast = True):
        input_ids, global_masks, attn_masks, mask_heads, mask_label_evidence, mask_label_answers, \
            mask_label_answer_span, mask_label_condition, qa_id = self.load_batch(data)
            
        if self.config.contrastive_learning:
            contrastive_input, contrastive_pairs = self.con_sampler.batch_generate(data)
            data[0], data[1], data[2] = contrastive_input
            # contrastive pairs is a list: [torch.Tensor, ...] where each tensor is like: torch.tensor([[1, 2], [37, 121], [133, 66]]) \
            # where each pair is the transformed HTMLElement index from A to B 
            # the enhanced tensors be like: [origin_1, enhanced_1, origin_2, enhanced_2]
            # the enhanced data only participate in contrastive learning part, not QA part

        input_ids, global_masks, attn_masks, mask_heads, mask_label_evidence, mask_label_answers, \
            mask_label_answer_span, mask_label_condition, qa_id = self.load_batch(data)
        if autocast == True:
            with torch.cuda.amp.autocast():
                last_hidden_all = self.transformer(input_ids, global_attention_mask = global_masks, attention_mask = attn_masks)
        else:
            last_hidden_all = self.transformer(input_ids, global_attention_mask = global_masks, attention_mask = attn_masks)
        if self.config.contrastive_learning:
            last_hidden = last_hidden_all[::2]
        else:
            last_hidden = last_hidden_all

        # span_loss do not need re-index operation
        pred_span = self.span_probe(last_hidden)[mask_label_answer_span != 0]
        label_span = (mask_label_answer_span[mask_label_answer_span != 0] + 1) / 2
        loss_span = self.weighted_loss(pred_span, label_span)

        # evidence loss / answersentence loss / condition_loss need re-index 
        # re-index process
        indexer = ReIndexer()
        indexer.set_index(mask_heads)
        max_head_num = indexer.get_max_head_num()
        mask_r = indexer.get_mask_r().to(input_ids.device)
        last_hidden_r = indexer.re_index(last_hidden)[:, :max_head_num] # B * HTML * H
        mask_label_evidence_r = indexer.re_index(mask_label_evidence)[:, :max_head_num]
        mask_label_answers_r = indexer.re_index(mask_label_answers)[:, :max_head_num]

        # predict evidence because it is before attention layer
        label_evidence = (mask_label_evidence_r[mask_label_evidence_r != 0] + 1) / 2
        
        pred_evidence = self.evidence_probe(last_hidden_r).squeeze(-1)[mask_label_evidence_r != 0]
        loss_evidence = self.weighted_loss(pred_evidence, label_evidence)

        # predict answer because it is after attention layer
        # last_hidden_r = self.head_attention(last_hidden_r, last_hidden_r, last_hidden_r, key_padding_mask = (~ mask_r))[0]
        mask_answers = mask_label_answers_r != 0
        label_answers = (mask_label_answers_r + 1) / 2
        pred_answers = self.answer_probe(last_hidden_r)
        label_yes = label_answers[:, 0][mask_answers[:, 0]]
        label_no = label_answers[:, 1][mask_answers[:, 1]]
        pred_yes = pred_answers[:, 0][mask_answers[:, 0]]
        pred_no = pred_answers[:, 1][mask_answers[:, 1]]
        label_extractive = label_answers[:, 2:][mask_answers[:, 2:]]
        pred_extractive = pred_answers[:, 2:][mask_answers[:, 2:]]

        loss_yesno = (self.weighted_loss(pred_yes, label_yes) + self.weighted_loss(pred_no, label_no)) /2
        loss_extractive = self.weighted_loss(pred_extractive, label_extractive)
        
        # predict condition and predict.
        pred_condition = self.condition_calculator(last_hidden_r, mask_r)
        # get label condition
        ans_indicator = mask_label_condition[..., 0]
        cond_indicator = mask_label_condition[..., 1]
        ans_indicator = indexer.re_index(ans_indicator)[:, :max_head_num].transpose(-2, -1)
        cond_indicator = indexer.re_index(cond_indicator)[:, :max_head_num].transpose(-2, -1)
        cond_indicator = cond_indicator.unsqueeze(2).repeat(1, 1, max_head_num, 1)# 1, 2, 3, 3
        label_condition = torch.einsum('abcd,abc->acd',cond_indicator, ans_indicator)# NOTE there might be some bugs because never see 1
        mask_r = torch.einsum('ab,ac->abc', mask_r, mask_r)
        mask_r = torch.einsum('abc,ab->abc', mask_r, (mask_label_answers_r + 1) / 2)
        pred_condition = pred_condition[mask_r!=0]
        label_condition = label_condition[mask_r!=0]
        loss_condition = self.weighted_loss(pred_condition, label_condition)

        loss_total = self.evidence_weight * loss_evidence + self.yesno_weight * loss_yesno + self.extractive_weight * loss_extractive + \
            self.span_weight * loss_span + self.condition_weight * loss_condition

        
        if self.config.contrastive_learning:
            last_hidden_resized = last_hidden_all.reshape(last_hidden_all.shape[0] // 2, 2, last_hidden_all.shape[1], last_hidden_all.shape[2])
            contrastive_loss = self.contrastive_loss(last_hidden_resized, contrastive_pairs)
            loss_total += self.contrastive_weight * contrastive_loss * (loss_evidence.detach() / contrastive_loss.detach()) # contrastive loss will be normalized
            return loss_total, to_numpy(loss_evidence, loss_yesno, loss_extractive, loss_span, loss_condition, contrastive_loss)

        return loss_total, to_numpy(loss_evidence, loss_yesno, loss_extractive, loss_span, loss_condition)


class HPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = HPTEncoder(config)
        self.mode = None
        self.lr_warmup_weight = min((config.epoch + 1) / config.warmup_epoch_num, 1)
        self.lr_decay_weight = min(1, (config.total_epoch_num - config.epoch) / (config.total_epoch_num - config.warmup_epoch_num))
        self.optimizer = torch.optim.AdamW([
            {'params':self.encoder.parameters(), 'lr':3e-5 * self.lr_warmup_weight * self.lr_decay_weight, 'weight_decay':0.01},
        ])
        self.optimizer.zero_grad()
        self.scaler = torch.cuda.amp.GradScaler()
        self.tokenizer = Tokenizer(config.model_root)

            
    def activate_training_mode(self):
        # train with DDP
        if self.mode != None:
            self.activate_normal_mode()
        try:
            self.to('cpu')
            self.cuda(self.config.local_rank)
            self.encoder = torch.nn.parallel.DistributedDataParallel(
                self.encoder, device_ids=[self.config.local_rank], output_device=self.config.local_rank, find_unused_parameters=True
                )
            self.encoder.module.transformer.train()
        except:
            self.to('cuda')
        self.mode = 'train'

    def activate_inference_mode(self):
        # inference with DP
        if self.mode != None:
            self.activate_normal_mode()
        self.to('cuda:0')
        self.encoder.transformer = torch.nn.DataParallel(self.encoder.transformer, device_ids=[0,1])
        self.encoder.eval()
        self.mode = 'eval'

    def activate_normal_mode(self):
        #non-paralleled mode
        if self.mode == 'train':
            self.encoder = self.encoder.module
        elif self.mode == 'eval':
            self.encoder.transformer = self.encoder.transformer.module
        self.encoder.eval()
        self.to('cpu')
        self.mode = None

    def train(self, train_inputs):
        self.optimizer.zero_grad()
        self.activate_training_mode()
        batch_steps = 0
        record_losses = []
        for i in range(self.config.repeat):
            if self.config.tqdm:
                iter = tqdm(train_inputs, total=len(train_inputs))
            else:
                iter = train_inputs
            for batch in iter:
                batch_steps += 1
                loss, rec_l = self.encoder(batch, autocast = True)
                record_losses.append(rec_l)
                loss /= self.config.accumulation_step
                self.scaler.scale(loss).backward()
                if (batch_steps + 1) % self.config.accumulation_step == 0:
                    self.scaler.unscale_(self.optimizer)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    logger2.log(torch.tensor(record_losses).nanmean(0))
        self.activate_normal_mode()
        torch.save(self.state_dict(), os.path.join(self.config.model_root, 'model_current.pt'))


    def test(self, dev_inputs):
        '''
        develop on the dev set to get best threshold for test set
        '''
        self.activate_inference_mode()

        preds = {'evidence': [], 'answers': [], 'condition': [], 'yes': [], 'no': [], 'extractive': []}
        labels = {'evidence': [], 'answers': [], 'condition': [], 'yes': [], 'no': [], 'extractive': []}
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                if self.config.tqdm:
                    dev_inputs = tqdm(dev_inputs, total=len(dev_inputs))
                for data in dev_inputs:
                    input_ids, global_masks, attn_masks, mask_heads, mask_label_evidence, mask_label_answers, \
                        mask_label_answer_span, mask_label_condition, qa_id = self.encoder.load_batch(data)

                    last_hidden = self.encoder.transformer(input_ids, global_attention_mask = global_masks, attention_mask = attn_masks)

                    indexer = ReIndexer()
                    indexer.set_index(mask_heads)
                    max_head_num = indexer.get_max_head_num()
                    mask_r = indexer.get_mask_r().to(input_ids.device)
                    last_hidden_r = indexer.re_index(last_hidden)[:, :max_head_num] # B * HTML * H
                    mask_label_evidence_r = indexer.re_index(mask_label_evidence)[:, :max_head_num]
                    mask_label_answers_r = indexer.re_index(mask_label_answers)[:, :max_head_num]
                    
                    # predict evidence because it is before attention layer
                    label_evidence = (mask_label_evidence_r[mask_label_evidence_r != 0] + 1) / 2
                    pred_evidence = self.encoder.evidence_probe(last_hidden_r).sigmoid().squeeze(-1)[mask_label_evidence_r != 0]

                    # predict answer because it is after attention layer
                    # last_hidden_r = self.encoder.head_attention(last_hidden_r, last_hidden_r, last_hidden_r, \
                    #     key_padding_mask = (~ mask_r))[0]

                    mask_answers = mask_label_answers_r != 0
                    label_answers = (mask_label_answers_r + 1) / 2
                    pred_answers = self.encoder.answer_probe(last_hidden_r).sigmoid().squeeze(-1)
                    label_yes = label_answers[:, 0][mask_answers[:, 0]]
                    label_no = label_answers[:, 1][mask_answers[:, 1]]
                    label_extractive = label_answers[:, 2:][mask_answers[:, 2:]]
                    pred_yes = pred_answers[:, 0][mask_answers[:, 0]]
                    pred_no = pred_answers[:, 1][mask_answers[:, 1]]
                    pred_extractive = pred_answers[:, 2:][mask_answers[:, 2:]]
                    label_answers = label_answers[mask_answers]
                    pred_answers = pred_answers[mask_answers]
                    

                    # predict condition and predict.
                    pred_condition = self.encoder.condition_calculator(last_hidden_r, mask_r).sigmoid()
                    ans_indicator = mask_label_condition[..., 0]
                    cond_indicator = mask_label_condition[..., 1]
                    ans_indicator = indexer.re_index(ans_indicator)[:, :max_head_num].transpose(-2, -1)
                    cond_indicator = indexer.re_index(cond_indicator)[:, :max_head_num].transpose(-2, -1)
                    cond_indicator = cond_indicator.unsqueeze(2).repeat(1, 1, max_head_num, 1)# 1, 2, 3, 3
                    label_condition = torch.einsum('abcd,abc->acd',cond_indicator, ans_indicator)# NOTE there might be some bugs because never see 1
                    mask_r = torch.einsum('ab,ac->abc', mask_r, mask_r)
                    mask_r = torch.einsum('abc,ab->abc', mask_r, (mask_label_answers_r + 1) / 2)
                    pred_condition = pred_condition[mask_r!=0]
                    label_condition = label_condition[mask_r!=0]

                    preds['evidence'] += pred_evidence.cpu().detach().tolist()
                    labels['evidence'] += label_evidence.cpu().detach().tolist()
                    preds['answers'] += pred_answers.cpu().detach().tolist()
                    labels['answers'] += label_answers.cpu().detach().tolist()
                    preds['yes'] += pred_yes.cpu().detach().tolist()
                    labels['yes'] += label_yes.cpu().detach().tolist()
                    preds['no'] += pred_no.cpu().detach().tolist()
                    labels['no'] += label_no.cpu().detach().tolist()
                    preds['extractive'] += pred_extractive.cpu().detach().tolist()
                    labels['extractive'] += label_extractive.cpu().detach().tolist()
                    preds['condition'] += pred_condition.cpu().detach().tolist()
                    labels['condition'] += label_condition.cpu().detach().tolist()
        thresholds = []
        for name in ['evidence', 'answers', 'yes', 'no', 'extractive', 'condition']:
            logger.log(name)
            if name in ['evidence']:
                weight = 0.1
            if name in ['answers', 'extractive']:
                weight = 1
            if name in ['yes', 'no']:
                weight = 1
            if name == 'condition':
                weight = 1
            fk, auc, threshold, p, r = analyze_binary_classification(labels[name], preds[name], prec_weight = weight)
            logger.log(f'f{weight}_score:%.4f, auc:%.4f, threshold for best_f1:%.4f, prec:%.4f, recall:%.4f'%\
                (fk, auc, threshold, p, r))

            thresholds.append(threshold)
        thresholds[2] = 0
        thresholds[3] = 0
        thresholds[4] = 1
        # thresholds[-1] = 1.0

        return thresholds

    def answering_questions(self, test_inputs, max_prediction):
        '''
        making inference on test set
        '''
        self.activate_inference_mode()

        softmax_layer = torch.nn.Softmax(dim = -1)
        thre_evid, thre_ans, thre_yes, thre_no, thre_extractive, thre_condition = self.test(test_inputs)
        logger.log('testing on test inputs')
        output = {}
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                if self.config.tqdm:
                    test_inputs = tqdm(test_inputs, total=len(test_inputs))
                for data in test_inputs:
                    input_ids, global_masks, attn_masks, mask_heads, mask_label_evidence, mask_label_answers, \
                        mask_label_answer_span, mask_label_condition, qa_id = self.encoder.load_batch(data)
                    last_hiddens = self.encoder.transformer(input_ids, global_attention_mask = global_masks, attention_mask = attn_masks)
                    sample_count = data[0].shape[0]
                    for sample_index in range(sample_count):
                        last_hidden = last_hiddens[sample_index]
                        text_ids = input_ids[sample_index].cuda(last_hidden.device)
                        index_heads = mask_heads[sample_index].nonzero().flatten()
                        id_example = qa_id[sample_index].tolist()

                        global_mask = global_masks[sample_index].nonzero().flatten()
                        index_end = global_mask[global_mask > index_heads.max()].min().unsqueeze(0)
                        range_answer_span = torch.concat([
                            index_heads.reshape(-1,1), 
                            torch.concat([index_heads[1:], index_end]).reshape(-1,1) - 1
                        ], dim = 1)
                        head_hiddens = last_hidden[index_heads]
                        pred_evidence = self.encoder.evidence_probe(head_hiddens).sigmoid().squeeze(-1)
                        # head_hiddens = self.encoder.head_attention(head_hiddens, head_hiddens, head_hiddens)[0]
                        head_hiddens = head_hiddens[None]
                        pred_answers = self.encoder.answer_probe(head_hiddens)[0].sigmoid().squeeze(-1)

                        pred_answers_mask = torch.concat(
                            [pred_answers[0:1] > thre_yes, pred_answers[1:2] > thre_no, pred_answers[2:] > thre_extractive]
                            )
                        # comment below when you need to predict answers
                        mask_label_answer = mask_label_answers[sample_index:sample_index+1]

                        pred_answers_mask = (mask_label_answer[mask_label_answer != 0] + 1) / 2 # only need to predict conditions
                        pred_answer_sentence_span = range_answer_span[pred_answers_mask.bool()]
                        prob_pred_answers = pred_answers[pred_answers_mask.bool()]

                        indexer = ReIndexer()
                        indexer.set_index(mask_heads[sample_index:sample_index+1])
                        max_head_num = indexer.get_max_head_num()
                        last_hidden_r = indexer.re_index(last_hidden.unsqueeze(0))[:, :max_head_num] # 1 * HTML * H
                        mask_r = indexer.get_mask_r().to(input_ids.device)
                        pred_condition = self.encoder.condition_calculator(last_hidden_r, mask_r)[0].sigmoid() # HTML * HTML
                        pred_condition = (pred_condition > thre_condition) & (pred_condition >= \
                            pred_condition.topk(min(5, pred_condition.shape[-1]), dim=-1).values[:, -1].unsqueeze(1))

                        for index_answer, range_ in enumerate(pred_answer_sentence_span):
                            pred_answer_span = softmax_layer(self.encoder.span_probe(last_hidden[range_[0]: range_[1] + 1, :]).sigmoid().T)
                            index_pred_answer_start = torch.argmax(pred_answer_span[0])
                            index_pred_answer_end = torch.argmax(pred_answer_span[1])
                            # comment below when you need to predict answers
                            sample_mask_label_span = (mask_label_answer_span[sample_index] + 1) / 2
                            if range_[0] == range_[1]:
                                index_pred_answer_start = 0
                                index_pred_answer_end = 0
                            else:
                                index_pred_answer_start = sample_mask_label_span[range_[0]:range_[1]+1, 0].nonzero()[0, 0]
                                index_pred_answer_end = sample_mask_label_span[range_[0]:range_[1]+1, 1].nonzero()[0, 0]

                            prob = prob_pred_answers[index_answer]
                            pred_answer = self.tokenizer.decode(text_ids[index_pred_answer_start + range_[0]: index_pred_answer_end + range_[0] + 1])
                            answer_sentence_index = (indexer.index[0] == range_[0]).nonzero()[0][0]
                            pred_condition_index = pred_condition[answer_sentence_index] # for testing
                            pred_condition_start_end = range_answer_span[pred_condition_index]
                            pred_conditions = []
                            for start, end in pred_condition_start_end:
                                pred_conditions.append(self.tokenizer.decode(text_ids[start: end + 1]))
                            if pred_answer != '':
                                if '<yes>' in pred_answer:#NOTE for condition test
                                    pred_answer = 'yes'
                                if '<no>' in pred_answer:
                                    pred_answer = 'no'
                                if id_example in output:
                                    if pred_answer not in [i[0][0] for i in output[id_example]]:
                                        output[id_example].append([[pred_answer, pred_conditions], prob])
                                    else:
                                        for i in output[id_example]:
                                            if i[0][0] == pred_answer:
                                                i[0][1] += pred_conditions
                                                i[0][1] = list(set(i[0][1]))
                                else:
                                    output.update({id_example:[[[pred_answer, pred_conditions], prob]]})


            output_real = []
            def answers_to_list(answers):
                answers.sort(key = lambda x:x[1], reverse = True)
                answers = [x[0] for x in answers[:max_prediction]]
                return answers
                
            for k, v in output.items():
                if self.config.inference_data == 'train_data':
                    output_real.append({'id':'train-'+str(k), 'answers':answers_to_list(v)})
                elif self.config.inference_data == 'dev_data':
                    output_real.append({'id':'dev-'+str(k), 'answers':answers_to_list(v)})
            answered = set()
            for x in output_real:
                answered.add(int(x['id'].split('-')[1]))
            if self.config.inference_data == 'train_data':
                all_qa = set(range(0, 2338))
                for k in all_qa - answered:
                    output_real.append({'id':'train-'+str(k), 'answers':[]})
            if self.config.inference_data == 'dev_data':
                all_qa = set(range(0, 285))
                for k in all_qa - answered:
                    output_real.append({'id':'dev-'+str(k), 'answers':[]})

            json.dump(output_real, open(self.config.output_file,'w'))
            if self.config.inference_data == 'train_data':
                A = evaluate(self.config.output_file, os.path.join(self.config.data_root, 'train.json'))
            elif self.config.inference_data == 'dev_data':
                A = evaluate(self.config.output_file, os.path.join(self.config.data_root, 'dev.json'))
            metric_now = A['total']['EM'] + A['total']['F1']
            logger.log('metric: ' + str(A))
            logger.log('total: %.6f'%(metric_now))
            
        return metric_now
