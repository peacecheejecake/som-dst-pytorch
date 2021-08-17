from config_utils import load_yaml_config, set_seed
from data_utils import (
    SomDSTDataset,
    DOMAIN_IDS, OPERATION_IDS,
    load_data, load_slot_meta, train_dev_split, preprocess, postprocess,
    cast_state, encode_state,
)
from train_utils import CosineAnnealingAfterWarmUpScheduler, convert_millisecond_to_str
from eval_utils import DSTEvaluator
from modeling import SomDST

import os
import pickle
import json
from copy import deepcopy
import argparse
import random
import math
import neptune

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.multiprocessing import set_start_method
from transformers import AutoTokenizer


def train(config, model, train_loader, dev_data, slot_meta):
    neptune.init(
        project_qualified_name='peace.cheejecake/stage3-DST',
        api_token=config.neptune_api_token,
    )
    neptune.create_experiment(config.experiment_name)
    
    if config.selective_decay:
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        enc_params_w_decay = []
        enc_params_wo_decay = []
        for name, param in model.opr_predictor.named_parameters():
            if any([nd in name for nd in no_decay]):
                enc_params_wo_decay.append(param)
            else:
                enc_params_w_decay.append(param)
        enc_param_groups = [
            {'params': enc_params_w_decay, 'weight_decay': config.weight_decay},
            {'params': enc_params_wo_decay, 'weight_decay': 0.},
        ]
        enc_optimizer = AdamW(
            params=enc_param_groups,
            lr=config.lr_enc.base,
            betas=config.betas,
            eps=config.eps,
        )
    else:
        raise NotImplementedError

    dec_optimizer = AdamW(
        params=list(model.val_generator.parameters()),
        lr=config.lr_dec.base,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )
    total_steps = config.epochs * (len(train_loader.dataset) // config.batch_size)
    enc_scheduler = CosineAnnealingAfterWarmUpScheduler(
        optimizer=enc_optimizer, 
        warmup_steps=total_steps * config.enc_warmup_ratio,
        cycle_steps=config.cycle_steps,
        max_lr=config.lr_enc.base,
        min_lr=config.lr_enc.min,
        damping_ratio=config.damping_ratio,
    )
    dec_scheduler = CosineAnnealingAfterWarmUpScheduler(
        optimizer=dec_optimizer, 
        warmup_steps=total_steps * config.dec_warmup_ratio,
        cycle_steps=config.cycle_steps,
        min_lr=config.lr_dec.min,
        max_lr=config.lr_dec.base,
        damping_ratio=config.damping_ratio,
    )

    #debug
    tokenizer = AutoTokenizer.from_pretrained(config.plm_name_or_path)

    model_checkpoint = os.path.join(config.checkpoint_dir, config.checkpoint_model_name)
    enc_scheduler_checkpoint = os.path.join(config.checkpoint_dir, config.checkpoint_sch_e_name)
    dec_scheduler_checkpoint = os.path.join(config.checkpoint_dir, config.checkpoint_sch_d_name)
    if config.resume_train:
        model.load_state_dict(torch.load(model_checkpoint, map_location=config.device))
        with open(enc_scheduler_checkpoint, 'rb') as f:
            enc_scheduler.load_state_dict(pickle.load(f))
        with open(dec_scheduler_checkpoint, 'rb') as f:
            dec_scheduler.load_state_dict(pickle.load(f))

    num_steps_per_epoch = math.ceil(len(train_loader.dataset) / config.batch_size)
    dev_evaluator = DSTEvaluator(slot_meta)
    best_score = {'epoch': 0, 'joint_goal_accuracy': 0, 'turn_slot_accuracy': 0, 'turn_slot_f1': 0}
    no_increase = 0
    for epoch in range(config.epochs):
        if config.stop_count is not None and no_increase > config.stop_count:
            break

        model.train()
        start_of_epoch = torch.cuda.Event(enable_timing=True)
        end_of_epoch = torch.cuda.Event(enable_timing=True)
        start_of_epoch.record()
        for step, data in enumerate(train_loader):
            (_, input_ids, segment_ids, input_masks, slot_positions,
            opr_labels, dom_labels, val_labels) = data

            teacher = val_labels if random.random() < 0.5 else None

            p_opr, p_dom, p_svg = model(input_ids=input_ids, 
                                        segment_ids=segment_ids, 
                                        input_masks=input_masks, 
                                        slot_positions=slot_positions, 
                                        opr_ids=opr_labels,
                                        teacher=teacher)

            loss_opr = model.loss_fn_opr(p_opr.view(-1, p_opr.size(-1)), opr_labels.view(-1)) # score: (B, J, O), label: (B, J)
            loss_dom = model.loss_fn_dom(p_dom, dom_labels).mean() # score: (B, J, M), label: (B, J, M)
            loss = loss_opr + loss_dom
            if p_svg.size(1) > 0:
                loss_svg = model.loss_fn_svg(p_svg, val_labels) # score: (B, J', K, V), label: (B, J', V)
                loss += loss_svg
            else:
                loss_svg = 0. # dummy for print

            #debug
            if step % 100 == 0:
                print(p_opr.ne(0).sum())
                print(p_opr.max(-1)[-1])
                # for x in p_svg.max(-1)[-1].tolist():
                #     print(tokenizer.convert_ids_to_tokens(x))

            model.zero_grad()
            loss.backward()
            enc_optimizer.step()
            dec_optimizer.step()
            enc_scheduler.step()
            dec_scheduler.step()

            end_of_epoch.record()
            torch.cuda.synchronize()
            duration = convert_millisecond_to_str(start_of_epoch.elapsed_time(end_of_epoch))

            neptune.log_metric('loss', loss)
            neptune.log_metric('loss_opr', loss_opr)
            neptune.log_metric('loss_dom', loss_dom)
            neptune.log_metric('loss_svg', loss_svg)
            neptune.log_metric('enc_lr', enc_optimizer.param_groups[0]['lr'])
            neptune.log_metric('dec_lr', dec_optimizer.param_groups[0]['lr'])
            
            print(f"\r[epoch {epoch:02d} ({(step + 1)/ num_steps_per_epoch * 100:.0f}%)]", 
                  f"{loss:.4f}, {loss_opr:.4f}, {loss_dom:.4f}, {loss_svg:.4f} ({duration})  ", end="")
        
        print()
            
        pred = eval(config, model, dev_data, slot_meta, evaluator=dev_evaluator)
        score = dev_evaluator.compute()
        print(pred[list(pred.keys())[1]])
        print(score)
        
        if score['joint_goal_accuracy'] > best_score['joint_goal_accuracy']:
            best_score['epoch'] = epoch
            best_score.update(score)

            torch.save(model.state_dict(), model_checkpoint)
            with open(enc_scheduler_checkpoint, 'wb') as f:
                pickle.dump(enc_scheduler.state_dict(), f, pickle.HIGHEST_PROTOCOL)
            with open(dec_scheduler_checkpoint, 'wb') as f:
                pickle.dump(dec_scheduler.state_dict(), f, pickle.HIGHEST_PROTOCOL)
            
            print("Updated checkpoint.")
            no_increase = 0
        else:
            no_increase += 1
    

    if os.path.exists(model_checkpoint):
        model.load_state_dict(torch.load(model_checkpoint))
        print(f"\nBEST MODEL: {best_score}\n")
    
    return model


def eval(config, model, data, slot_meta, tokenizer=None, evaluator=None, load_model=False):
    if load_model:
        model_checkpoint = os.path.join(config.checkpoint_dir, config.checkpoint_model_name)
        model.load_state_dict(torch.load(model_checkpoint, map_location=config.device))
        print(f"Model state dict from: {model_checkpoint}")

    if tokenizer is None:
        tokenizer = model.tokenizer
    
    CLS = tokenizer.cls_token_id
    SEP = tokenizer.sep_token_id
    PAD = tokenizer.pad_token_id
    SLOT = tokenizer.convert_tokens_to_ids('[SLOT]')

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    starter.record()
    
    model.eval()
    predictions = {}
    for i, dialogue in enumerate(data):
        dial_idx = dialogue['dialogue_idx']
        last_state = {}
        last_utterances = [SEP]
        sys_response = ""
        for turn_idx, turn in enumerate(dialogue['dialogue']):
            if turn['role'] != 'user':
                sys_response = turn['text']
                continue
            
            idx = f"{dial_idx}-{turn_idx}"
            last_state = cast_state(slot_meta, last_state)
            last_state_ = encode_state(tokenizer, last_state, slot_meta)
            utterances = tokenizer.encode(f"{sys_response} ; {turn['text']} [SEP]",
                                          add_special_tokens=False)

            state_len = sum(map(len, last_state_)) + len(slot_meta)
            input_len = len(last_utterances) + len(utterances) + state_len
            if input_len > config.max_seq_len - 1:
                if input_len - len(last_utterances) < config.max_seq_len - 1:
                    available_len = config.max_seq_len - state_len - len(utterances) - 1
                    last_utterances = last_utterances[-available_len:]
                    assert SEP in last_utterances
                elif state_len < config.max_seq_len - 2:
                    available_len = config.max_seq_len - state_len - 2
                    last_utterances = [SEP]
                    utterances = utterances[-available_len:]
                    assert SEP in utterances
                else:
                    print(len(last_utterances), len(utterances), state_len)
                    print(last_state_)
                    raise ValueError("Too small max_seq_length.")

            input_len = len(last_utterances) + len(utterances) + state_len + 1
            pad_len = config.max_seq_len - input_len

            input_ids = [CLS] + last_utterances + utterances
            slot_positions = []
            for sv in last_state_:
                slot_positions.append(len(input_ids))
                input_ids.extend([SLOT] + sv)
            input_ids += [PAD] * pad_len
            segment_ids = (
                    [0] * (len(last_utterances) + 1)
                  + [1] * (len(utterances) + state_len)
                  + [0] * pad_len
            )
            input_masks = [1] * input_len + [0] * pad_len

            input_ids = torch.LongTensor(input_ids).to(config.device).unsqueeze(0)
            segment_ids = torch.LongTensor(segment_ids).to(config.device).unsqueeze(0)
            input_masks = torch.LongTensor(input_masks).to(config.device).unsqueeze(0)
            slot_positions = torch.LongTensor(slot_positions).to(config.device).unsqueeze(0)
            
            # print(input_ids.shape, segment_ids.shape, input_masks.shape, slot_positions.shape)
            with torch.no_grad():
                p_opr, _, p_svg = model(input_ids=input_ids, 
                                        segment_ids=segment_ids, 
                                        input_masks=input_masks, 
                                        slot_positions=slot_positions)

            pred_opr = p_opr.max(-1)[-1].squeeze(0).detach().tolist() # (J)
            if p_svg.size(1) > 0:
                pred_svg = p_svg.max(-1)[-1].squeeze(0).detach().tolist() # (J', K) or None
            else:
                pred_svg = []
                # #debug
                # print(pred_svg)


            _pred = postprocess(tokenizer, slot_meta, config.opr_code, pred_opr, pred_svg, last_state)
            pred = [f"{s}-{v}" for s, v in _pred.items()]
            predictions[idx] = pred

            if evaluator is not None:
                gold = turn['state']
                evaluator.update(gold, pred)

            # #debug
            # if evaluator is not None:
            #     print(gold, pred)
            # else:
            #     print(pred)
                
            last_state = _pred
            last_utterances = utterances

            ender.record()
            torch.cuda.synchronize()
            duration = convert_millisecond_to_str(starter.elapsed_time(ender))
            print(f"\rEvaluating({(i + 1) / len(data) * 100:.0f}%) {duration}", end="")
    
    print()

    return predictions
  

def submit(predictions, dirname, prefix):
    try:
        sub_list = sorted([f for f in os.listdir(dirname) if f.startswith(prefix)])
        sub_num = int(sub_list[-1].split('.')[0][-3:]) + 1
    except IndexError:
        sub_num = 0

    new_submit_path = os.path.join(dirname, f"{prefix}{sub_num:03d}.csv")
    with open(new_submit_path, 'w') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"Saved submission: {new_submit_path}")


def main(config, do_train, do_submit):
    opr2id = OPERATION_IDS[config.opr_code]

    # path check
    paths = [config.train_path, config.eval_path, config.submission_dir, config.checkpoint_dir]
    for path in paths:
        if path is not None and not os.path.exists(path):
            raise ValueError(f"{path} not exists.")
    
    additional_special_tokens = ['[NULL]', '[SLOT]', '[EOS]']
    tokenizer = AutoTokenizer.from_pretrained(config.plm_name_or_path)
    tokenizer.add_tokens(additional_special_tokens, special_tokens=True)
    slot_meta = load_slot_meta()

    model = SomDST(
        plm_name_or_path=config.plm_name_or_path,
        num_domains=len(DOMAIN_IDS),
        num_operations=len(opr2id),
        tokenizer=tokenizer,
        update_id=opr2id['update'],
        max_dec_len=config.max_val_len,
        dropout_rate=config.dropout_rate,
    ).to(config.device)

    # train
    if do_train:
        # data loading
        train_raw = load_data(config.train_path)
        if config.exp_starter:
            train_raw = random.sample(train_raw, 100)
        train_raw, dev_raw = train_dev_split(train_raw, config.valid_ratio)
        
        # #debug
        # dev_eval = DSTEvaluator(slot_meta)
        # eval(config, model, dev_raw, slot_meta, tokenizer, evaluator=dev_eval)
        # return

        train_data = preprocess(
            data=train_raw,
            tokenizer=tokenizer,
            max_seq_len=config.max_seq_len,
            opr_code=config.opr_code
        )
        train_set = SomDSTDataset(
            data=train_data,
            tokenizer=tokenizer,
            device=config.device,
            max_seq_len=config.max_seq_len,
            max_val_len=config.max_val_len,
            shuffle_state=False,
            word_dropout=True,
        )
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=False,
            collate_fn=train_set.collate_fn,
        )
        print("Loaded train & dev data.")
        model = train(config, model, train_loader, dev_raw, slot_meta)

    # infer eval set & create submission
    if do_submit:
        test_raw = load_data(config.eval_path)
        pred = eval(config, model, test_raw, slot_meta, tokenizer=tokenizer, load_model=True)
        if pred is not None:
            submit(pred, config.submission_dir, 'submission')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()#argument_default="--train")
    argparser.add_argument("--config", type=str, default=f"{os.path.join(os.path.dirname(__file__), 'config.yaml')}")
    argparser.add_argument("--train", action='store_true')
    # argparser.add_argument("--valid", action='store_true')
    argparser.add_argument("--eval", default=True, action='store_true')

    args = argparser.parse_args()
    config = load_yaml_config(args.config)
    set_seed(config.seed)
    if config.num_workers > 0:
        set_start_method('spawn')
    
    main(config, args.train, args.eval)
    
    # if args.train:
    #     train(config)
    # if args.valid:
    #     valid(config)
    # if args.eval:
    #     eval(config)