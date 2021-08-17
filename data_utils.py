import torch
from torch.utils.data import Dataset

import random
import json
import os
from collections import defaultdict
from itertools import combinations
from dataclasses import dataclass, fields
from copy import deepcopy
import numpy as np

import torch.nn.functional as F

from typing import List, Dict




def make_domain_ids(single_domains):
    single_domains.sort()
    domains = (
        [''] +
        single_domains +
        ['+'.join(dom) for dom in combinations(single_domains, 2)] +
        ['+'.join(dom) for dom in combinations(single_domains, 3)]
    )
    dom2id = {dom: i for i, dom in enumerate(domains)}
    print(domains)
    return dom2id


OPERATION_IDS = {
    'full': {'carryover': 0, 'update': 1, 'delete': 2, 'dontcare': 3, 'yes': 4, 'no': 5},
    'origin': {'carryover': 0, 'update': 1, 'delete': 2, 'dontcare': 3},
    'wo_del': {'carryover': 0, 'update': 1, 'dontcare': 2, 'yes': 3, 'no': 4}
}
SINGLE_DOMAINS = ['관광', '숙소', '식당', '지하철', '택시']
# DOMAIN_IDS = make_domain_ids(SINGLE_DOMAINS)
DOMAIN_IDS = {dom: i for i, dom in enumerate(SINGLE_DOMAINS)}


class SomDSTDataset(Dataset):
    def __init__(self, 
                 data, 
                 tokenizer, 
                 device, 
                 max_seq_len, 
                 max_val_len, 
                 shuffle_state, 
                 word_dropout):
        super(SomDSTDataset, self).__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.device = device
        self.shuffle_state = shuffle_state
        self.max_seq_len = max_seq_len
        self.max_val_len = max_val_len
        self.word_dropout = word_dropout
        self.pad_token_id = tokenizer.pad_token_id
        self.unk_token_id = tokenizer.unk_token_id
        self.slot_token_id = tokenizer.convert_tokens_to_ids('[SLOT]')
        

    def __getitem__(self, idx):
        if self.shuffle_state and random.random() < 0.5:
            self.data[idx].shuffle_state()
        return self.data[idx]


    def __len__(self):
        return len(self.data)


    def collate_fn(self, batch):
        indices = []
        input_ids = []
        segment_ids = []
        input_masks = []
        slot_positions = []
        opr_labels = []
        dom_labels = []
        val_labels = []

        tokens_to_ignore = self.tokenizer.convert_tokens_to_ids([';', '[CLS]', '[SEP]'])
        max_num_val = max(map(lambda b: len(b.val_labels), batch))
        dummy_val_label = [self.pad_token_id] * self.max_val_len

        UNK = self.unk_token_id
        CLS = self.tokenizer.cls_token_id
        PAD = self.pad_token_id
        SLOT = self.slot_token_id
        

        for b in batch:
            # word dropout
            if self.word_dropout:
                dialogue_ = [UNK 
                             if i not in tokens_to_ignore and random.random() < 0.1 
                             else i 
                             for i in b.dialogue]
            else:
                dialogue_ = b.dialogue

            input_id = [CLS] + dialogue_
            slot_position = []
            for sv in b.last_state:
                slot_position.append(len(input_id))
                input_id += [SLOT] + sv

            # padding
            if len(input_id) < self.max_seq_len:
                pad_len = self.max_seq_len - len(input_id)
                input_id.extend([PAD] * pad_len)

            assert len(input_id) == self.max_seq_len
            
            # convert & pad value labels
            _val_labels = []
            for value in b.val_labels.values():
                value = self.tokenizer.convert_tokens_to_ids(value)
                assert len(value) <= self.max_val_len, f"len(val)[{len(value)}] > max_val_len[{self.max_val_len}]"
                if len(value) < self.max_val_len:
                    gap = self.max_val_len - len(value)
                    value += [self.pad_token_id] * gap
                _val_labels.append(value)
            if len(_val_labels) < max_num_val:
                num_val_gap = max_num_val - len(b.val_labels)
                _val_labels.extend([dummy_val_label for _ in range(num_val_gap)])
            
            indices.append(b.idx)
            input_ids.append(input_id)
            segment_ids.append(b.segment_ids)
            input_masks.append(b.input_masks)
            slot_positions.append(slot_position)
            opr_labels.append(b.opr_labels)
            dom_labels.append(b.dom_labels.unsqueeze(0))
            val_labels.append(_val_labels)

        indices = torch.LongTensor(indices).to(self.device) # (B)
        input_ids = torch.LongTensor(input_ids).to(self.device) # (B, X)
        segment_ids = torch.LongTensor(segment_ids).to(self.device) # (B, X)
        input_masks = torch.LongTensor(input_masks).to(self.device) # (B, X)
        slot_positions = torch.LongTensor(slot_positions).to(self.device) # (B, J)
        opr_labels = torch.LongTensor(opr_labels).to(self.device) # (B, J)
        dom_labels = torch.cat(dom_labels).to(self.device) # (B, M)
        val_labels = torch.LongTensor(val_labels).to(self.device) # (B, J', K)

        assert val_labels.size(1) == opr_labels.eq(1).sum(-1).max(), \
            f"{val_labels}, {opr_labels.eq(1).sum(-1).max()}"
        
        return (indices, input_ids, segment_ids, input_masks, slot_positions,
                opr_labels, dom_labels, val_labels)


@dataclass
class SomDSTData(dict):
    idx: int
    dialogue: List[int] # (?); encoded [last_dialog, dialog]
    last_state: List[List[int]] # (J, ?); encoded
    segment_ids: List[int] # -> token_type_ids
    input_masks: List[int] # -> attention_masks
    opr_labels: List[List[int]] # (J, O); all state operations
    dom_labels: List[List[int]] # (J, M); one-hot domain
    val_labels: Dict[str, List[str]] # slot(J') -> tokeninzed val 
                                    # (needs convesion & padding in `collate_fn`)
    slot_meta: List[str] # (J)
    

    def __post_init__(self):
        for field in fields(self):
            k = field.name
            self[k] = getattr(self, k)


    def __getitem__(self, idx):
        if isinstance(idx, str):
            inner_dict = {k: v for k, v in self.items()}
            return inner_dict[idx]
        else:
            self.to_tuple()[idx]


    def to_tuple(self):
        return tuple(self[k] for k in self.keys())


    def shuffle_state(self):
        tmp_gen = []
        dummy = []
        for slot in self.slot_meta:
            val = self.val_labels.get(slot)
            if val is None:
                tmp_gen.append(dummy)
            else:
                tmp_gen.append(val)

        tmp = list(zip(self.slot_meta, self.last_state, self.opr_labels, tmp_gen))
        random.shuffle(tmp)
        
        self.slot_meta, self.last_state, self.opr_labels, tmp_gen = map(list, zip(*tmp))
        self.val_labels = {s: v for s, v in zip(self.slot_meta, tmp_gen) if v != dummy}



def load_data(datapath: os.PathLike):
    with open(datapath) as f:
        data = json.load(f)
    return data


def load_ontology(filepath: os.PathLike = None):
    if filepath is None:
        filepath = '/opt/ml/input/data/train/ontology.json'
    with open(filepath) as f:
        ontology = json.load(f)
    return ontology


def load_slot_meta(filepath: os.PathLike = None):
    if filepath is None:
        filepath = '/opt/ml/input/data/train/slot_meta.json'
    with open(filepath) as f:
        slot_meta = json.load(f)
    return slot_meta



def train_dev_split(raw_data, dev_ratio):
    dom_mapper = defaultdict(list)
    for dialogue in raw_data:
        domains = dialogue['domains']
        dom_mapper[len(domains)].append(dialogue['dialogue_idx'])

    dev_indices = []
    for indices in dom_mapper.values():
        num_to_pick = int(len(indices) * dev_ratio)
        picked = random.sample(indices, num_to_pick)
        dev_indices.extend(picked)

    train, dev = [], []
    for dialogue in raw_data:
        if dialogue['dialogue_idx'] in dev_indices:
            dev.append(dialogue)
        else:
            train.append(dialogue)

    return train, dev


def encode_state(tokenizer, state, slot_meta=None):
    if isinstance(state, dict):
        sv_generator = state.items() # (s, v) for all slots
    else:
        slot_meta = load_slot_meta() if slot_meta is None else slot_meta
        sv_generator = zip(slot_meta, state)
    
    state_ids = []
    for slot, value in sv_generator:
        slot = ' '.join(slot.split('-'))
        sv = f"{slot} - {value} "
        state_ids.append(tokenizer.encode(sv, add_special_tokens=False))
    return state_ids


def cast_state(slot_meta, state):
    state_ = []
    for slot in slot_meta:
        value = state.get(slot)
        if value is None:
            state_.append('[NULL]')
        else:
            state_.append(value)
    return state_


def extract_info(data, slot_meta, tokenizer, opr_code):
    guids = []
    indices = []
    last_states = []
    last_dialogues = []
    states = []
    dialogues = []
    operations = []
    turn_domains = []
    gen_values = []
    
    idx = 0
    for dialogue in data:
        dial_id = dialogue['dialogue_idx']
        last_state = {}
        last_utterances = ""
        sys_response = ""
        for turn_idx, turn in enumerate(dialogue['dialogue']):
            if turn['role'] == 'user':
                guid = f"{dial_id}-{turn_idx}"
                state = fix_state(turn['state'])
                operation, gen_value = get_operation_and_gen_value(state, last_state, tokenizer, slot_meta, opr_code)
                turn_domain = encode_turn_domain(state, last_state)
                user_utterance = turn['text']
                utterances = f"{sys_response} ; {user_utterance} [SEP]"

                guids.append(guid)
                indices.append(idx)
                last_states.append(last_state)
                last_dialogues.append(last_utterances)
                states.append(state)
                dialogues.append(utterances)
                operations.append(operation)
                turn_domains.append(turn_domain)
                gen_values.append(gen_value)
                
                idx += 1
                last_state = state
                last_utterances = utterances
            else:
                sys_response = turn['text']

    return guids, indices, last_states, last_dialogues, states, dialogues, operations, turn_domains, gen_values


def encode_turn_domain(state, last_state):
    turn_domain = set()
    for slot, value in state.items():
        dom = slot.split('-')[0]
        if last_state.get(slot) != value:
            turn_domain.add(dom)
    for slot, value in last_state.items():
        if state.get(slot) != value:
            turn_domain.add(dom)
    domain_one_hot = [1 if dom in turn_domain else 0 for dom in DOMAIN_IDS]
    num_domains = sum(domain_one_hot)
    if num_domains > 1:
        domain = [o / num_domains for o in domain_one_hot]
    else:
        domain = domain_one_hot
    return F.softmax(torch.FloatTensor(domain), dim=-1)
    # turn_domain = sorted(turn_domain)
    
    # return DOMAIN_IDS['+'.join(turn_domain)]


def fix_state(state):
    state_ = {}
    for sv in state:
        s, v = sv.rsplit('-', 1)
        if v == 'dontcare':
            state_[s] = ''
        elif v == 'yes':
            state_[s] = '응'
        elif v == 'no':
            state_[s] = '아니'
        else:
            state_[s] = v
    return state_


def recover_state(slot_meta, state):
    state_ = {}
    for s, v in zip(slot_meta, state):
        if v == '상관 없음':
            state_[s] = 'dontcare'
        elif v == '응':
            state_[s] = 'yes'
        elif v == '아니':
            state_[s] = 'no'
        elif v != '[NULL]':
            state_[s] = v
    return state_


def get_operation_and_gen_value(state, last_state, tokenizer, slot_meta, opr_code):
    opr2id = OPERATION_IDS[opr_code]
    operations = [opr2id['carryover']] * len(slot_meta)
    gen_values = {}
    for idx, slot in enumerate(slot_meta):
        value = state.get(slot)
        last_value = last_state.get(slot)
        if value != last_value:
            if value is None and opr2id.get('delete') is not None:
                operations[idx] = opr2id['delete']
            elif value == '상관 없음' and opr2id.get('dontcare') is not None:
                operations[idx] = opr2id['dontcare']
            elif value == '응' and opr2id.get('yes') is not None:
                operations[idx] = opr2id['yes']
            elif value == '아니' and opr2id.get('no') is not None:
                operations[idx] = opr2id['no']
            else:
                operations[idx] = opr2id['update']
                value = tokenizer.tokenize(value) + ['[EOS]']
                gen_values[slot] = value

    return operations, gen_values


def encode_data(dialogues, last_dialogues, last_states, slot_meta, tokenizer, max_seq_len):
    CLS = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
    SEP = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)

    dialogues_ = []
    last_states_ = []
    segment_ids = []
    input_masks = []

    for idx, last_state in enumerate(last_states):
        last_state = cast_state(slot_meta, last_state)
        last_dialogue = last_dialogues[idx]
        dialogue = dialogues[idx]

        last_state = encode_state(tokenizer, last_state, slot_meta)
        last_dialogue = tokenizer.encode(last_dialogue, add_special_tokens=False)
        dialogue = tokenizer.encode(dialogue, add_special_tokens=False)

        state_len = sum(map(len, last_state)) + len(slot_meta)
        input_len = state_len + len(last_dialogue) + len(dialogue)
        if input_len > max_seq_len - 1:
            if input_len - len(last_dialogue) < max_seq_len - 1:
                available_len = max_seq_len - state_len - len(dialogue) - 1
                last_dialogue = last_dialogue[-available_len:]
                assert SEP in last_dialogue
            elif state_len < max_seq_len - 2:
                available_len = max_seq_len - state_len - 2
                last_dialogue = [SEP]
                dialogue = dialogue[-available_len:]
                assert SEP in dialogue
            else:
                print(state_len, len(last_dialogue), len(dialogue))
                raise ValueError("Too small max_seq_length.")
            input_len = len(last_dialogue) + len(dialogue) + state_len + 1
            pad_len = 0
            # segment_id = [0] * len(last_dialogue) + [1] * (len(dialogue) + state_len)
            # input_mask = [1] * max_seq_len
        else:
            input_len += 1
            pad_len = max_seq_len - input_len
        
        segment_id = [0] * (len(last_dialogue) + 1) + [1] * (len(dialogue) + state_len) + [0] * pad_len
        input_mask = [1] * input_len + [0] * pad_len

        assert len(segment_id) == max_seq_len and len(input_mask) == max_seq_len, \
            f"{len(segment_id)}, {len(input_mask)}, {input_len}, {pad_len}"

        dialogues_.append(last_dialogue + dialogue)
        last_states_.append(last_state)
        segment_ids.append(segment_id)
        input_masks.append(input_mask)
    
    return dialogues_, last_states_, segment_ids, input_masks


def pack_data(indices, slot_meta, *data):
    packed_data = []
    (dialogues, last_states, segment_ids, input_masks,
    operations, turn_domains, gen_ids) = data
    for idx in indices:
        packed_data.append(SomDSTData(idx=idx,
                                      dialogue=dialogues[idx],
                                      last_state=last_states[idx],
                                      segment_ids=segment_ids[idx],
                                      input_masks=input_masks[idx],
                                      opr_labels=operations[idx],
                                      dom_labels=turn_domains[idx],
                                      val_labels=gen_ids[idx],
                                      slot_meta=slot_meta))
    return packed_data


def preprocess(data, tokenizer, max_seq_len, opr_code, slot_meta_path=None):
    data = load_data(data) if isinstance(data, str) else data
    slot_meta = load_slot_meta(slot_meta_path)

    guids, indices, last_states, last_dialogues, states, dialogues, \
        operations, turn_domains, gen_ids = extract_info(data, slot_meta, tokenizer, opr_code)
    dialogues, last_states, segment_ids, input_masks = encode_data(
        dialogues, last_dialogues, last_states, slot_meta, tokenizer, max_seq_len
    )
    data = pack_data(indices, slot_meta, dialogues, last_states, 
                     segment_ids, input_masks, operations, turn_domains, gen_ids)
    # labels = {guid: state for guid, state in zip(guids, states)}

    return data#, labels


def postprocess(tokenizer, slot_meta, opr_code, pred_opr, pred_svg, last_state, 
                tokens_to_ignore=None):
    r"""
    Args:
        tokenizer: PretrainedTokenizer
        perd_opr: List[int]; (J)
        pred_svg: List[List[int]]; (J', K)
        last_state: List[str] - J `slot-value` pairs.
    """
    if tokens_to_ignore is None:
        tokens_to_ignore = [';', '-']

    opr2id = OPERATION_IDS[opr_code]
    id2opr = {v: k for k, v in opr2id.items()}
    pred_opr = [id2opr[i] for i in pred_opr]
    last_state = recover_state(slot_meta, last_state) # cast(list) -> not cast(dict)

    pred_svg = [tokenizer.convert_ids_to_tokens(i) for i in pred_svg]
    # #debug
    # print('post', pred_svg)

    pred = {}
    for slot, opr in zip(slot_meta, pred_opr):
        if opr == 'dontcare' and opr2id.get(opr) is not None:
            pred[slot] = 'dontcare'
        elif opr == 'yes' and opr2id.get(opr) is not None:
            pred[slot] = 'yes'
        elif opr == 'no' and opr2id.get(opr) is not None:
            pred[slot] = 'no'
        elif opr == 'delete' and opr2id.get(opr) is not None and last_state.get(slot) is not None:
            continue # none
        elif opr == 'update':
            tokens = pred_svg.pop(0)
            gen_value = []
            for token in tokens:
                if token == '[EOS]':
                    break
                elif token in tokens_to_ignore:
                    continue
                # else:
                #     gen_value.append(token)
                gen_value.append(token)
            
            gen_value = ' '.join(gen_value)
            gen_value = gen_value.replace(' ##', '').replace(' : ', ':')
            if gen_value != '[NULL]':
                pred[slot] = gen_value
        elif last_state.get(slot) is not None: #carryover
            pred[slot] = last_state[slot]

    return pred
