from collections import OrderedDict
from multiprocessing import Value

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, ElectraModel, PreTrainedTokenizerBase



class SomDST(nn.Module):
    def __init__(
        self,
        plm_name_or_path: str,
        num_domains: int,
        num_operations: int,
        tokenizer: PreTrainedTokenizerBase,
        update_id: int,
        max_dec_len: int,
        max_update: int = None,
        dropout_rate: int = 0.1,
    ):
        super(SomDST, self).__init__()
        self.update_id = update_id
        self.max_dec_len = max_dec_len
        self.max_update = max_update
        self.vocab_size = len(tokenizer)
        self.tokenizer = tokenizer
        self.padding_idx = self.tokenizer.pad_token_id

        self.opr_predictor = StateOperationPredictor(
            plm_name_or_path=plm_name_or_path,
            num_domains=num_domains,
            num_operations=num_operations,
            dropout_rate=dropout_rate,
        )
        self.reinit_special_token_embeddings()
        self.hidden_size = self.opr_predictor.encoder.config.hidden_size

        self.val_generator = SlotValueGenerator(
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            padding_idx=self.padding_idx,
            update_id=update_id,
            dropout_rate=dropout_rate,
        )
        self.tie_embeddings()
        self.reinit_rnn_weights()

        self.loss_fn_opr = nn.CrossEntropyLoss()
        # self.loss_fn_dom = CrossEntropyLossForDomain()
        self.loss_fn_dom = nn.CosineSimilarity()
        self.loss_fn_svg = MaskedCrossEntropyLossForValue(padding_idx=tokenizer.pad_token_id)

    def forward(self, 
                input_ids, 
                segment_ids, 
                input_masks, 
                slot_positions, 
                opr_ids=None,
                teacher=None):
        """
        X : input length,
        B : batch size,
        D : hidden size,
        M : # of domains,
        J : # of slots,
        J': # of update,
        O : # of operation,
        K : max value length,
        """
        device = input_ids.device

        sequence_output, hidden_dom, hidden_opr, p_dom, p_opr = self.opr_predictor(
            input_ids=input_ids,
            segment_ids=segment_ids,
            input_masks=input_masks,
            slot_positions=slot_positions,
        )

        if opr_ids is None:
            opr_ids = p_opr.max(-1)[-1] # (B, J)

        # #debug
        # print()
        # print(input_ids, slot_positions)

        if self.max_update is not None:
            max_update = self.max_update
        else:
            max_update = opr_ids.eq(self.update_id).sum(-1).max().item()

        decoder_input = []
        for b, opr_mask in zip(hidden_opr, opr_ids.eq(self.update_id)): # b: (J, D), opr_mask: (J) -> (J')
            if opr_mask.any():
                to_update = b[opr_mask].unsqueeze(0) # (1, J', D)
                gap = max_update - to_update.size(1)
                assert gap >= 0
                if gap > 0:
                    pads = torch.zeros((1, gap, self.hidden_size), device=device)
                    to_update = torch.cat([to_update, pads], dim=1)
            else:
                to_update = torch.zeros((1, max_update, self.hidden_size), device=device) #dummy
            decoder_input.append(to_update)
        decoder_input = torch.cat(decoder_input) # (B, J', D)

        p_svg = self.val_generator(input_ids=input_ids,
                                    decoder_input=decoder_input,
                                    hidden=hidden_dom.unsqueeze(0),
                                    encoder_output=sequence_output,
                                    max_len=self.max_dec_len,
                                    teacher=teacher)

        return p_opr, p_dom, p_svg

    def tie_embeddings(self):
        self.val_generator.embeddings.weight = self.opr_predictor.encoder.embeddings.word_embeddings.weight

    def reinit_special_token_embeddings(self):
        self.opr_predictor.encoder.resize_token_embeddings(self.vocab_size)
        for token in self.tokenizer.additional_special_tokens:
            token_id = self.tokenizer.convert_token_to_id(token)
            self.opr_predictor.encoder.embeddings.word_embeddings.weight.data[token_id]\
                .normal_(mean=0., std=self.opr_predictor.enconder.config.initializer_range)

    def reinit_rnn_weights(self):
        for name, param in self.val_generator.decoder.named_parameters():
            if 'weight' in name:
                param.data.normal_(mean=0.0, std=0.02)


class StateOperationPredictor(nn.Module):
    def __init__(self, plm_name_or_path, num_domains, num_operations, dropout_rate):
        super(StateOperationPredictor, self).__init__()
        self.encoder = BertModel.from_pretrained(plm_name_or_path)
        self.hidden_size = self.encoder.config.hidden_size
        self.num_domains = num_domains
        self.num_oprs = num_operations

        self.dom_classifier = nn.Sequential(OrderedDict({
            'dropout': nn.Dropout(dropout_rate),
            'linear_in': nn.Linear(self.hidden_size, self.hidden_size),
            'relu': nn.ReLU(),
            'linear_out': nn.Linear(self.hidden_size, self.num_domains),
            'softmax': nn.Softmax(dim=-1)
        }))
        self.opr_classifier = nn.Sequential(OrderedDict({
            'dropout': nn.Dropout(dropout_rate),
            'linear_in': nn.Linear(self.hidden_size, self.hidden_size),
            'relu': nn.ReLU(),
            'linear_out': nn.Linear(self.hidden_size, self.num_oprs),
            'softmax': nn.Softmax(dim=-1),
        }))
        self.pooler = nn.Sequential(OrderedDict({
            'linear': nn.Linear(self.hidden_size, self.hidden_size),
            'tanh': nn.Tanh(),
        }))

    def forward(self, input_ids, segment_ids, input_masks, slot_positions):
        """
        X : input length,
        B : batch size,
        D : hidden size,
        M : # of domains,
        J : # of slots,
        J': # of update,
        O : # of operation,
        K : max value length,
        """

        r"""
        Args:
            input_ids: (B, X)
            segment_ids: (B, X)
            input_masks: (B, X)
            slot_positions: (B, J)
        
        Returns:
            sequence_output: (B, X, D)
            pooler_output: (B, D)
            hidden_opr: (B, J, D)
            p_dom: (B, J, M)
            p_opr: (B, J, O)
        """

        bert_output = self.encoder(
            input_ids=input_ids,
            token_type_ids=segment_ids,
            attention_mask=input_masks,
            return_dict=False,
        )
        if isinstance(self.encoder, BertModel):
            sequence_output, pooler_output = bert_output[:2] # seq. output: (B, X, D), pooler: (B, D)
        elif isinstance(self.encoder, ElectraModel):
            sequence_output = bert_output[0]
            pooler_output = self.pooler(sequence_output[:, 0, :])
        else:
            raise NotImplementedError

        p_dom = self.dom_classifier(pooler_output) # (B, M)
        slot_positions = slot_positions.unsqueeze(-1).expand(-1, -1, self.hidden_size) # (B, J, D)
        hidden_opr = torch.gather(sequence_output, dim=1, index=slot_positions) # (B, J, D)
        # print('hidden opr:', hidden_opr)
        # print('wieght:', self.opr_classifier.linear.weight.transpose(0, 1))
        p_opr = self.opr_classifier(hidden_opr) # (B, J, O)
        # print('p_opr:', p_opr)
        # raise OSError
        # p_opr[:, :, 0] /= 33.

        return sequence_output, pooler_output, hidden_opr, p_dom, p_opr


class SlotValueGenerator(nn.Module):
    def __init__(self, 
                 hidden_size, 
                 vocab_size, 
                 update_id, 
                 dropout_rate,
                 padding_idx=None): 
                 
        super(SlotValueGenerator, self).__init__()
        self.decoder = nn.GRU(input_size=hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              batch_first=True)
        self.embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)
        self.linear_gen = nn.Linear(3 * hidden_size, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()
        self.padding_idx = padding_idx if padding_idx is not None else 0
        self.update_id = update_id
        self.vocab_size = vocab_size

    def forward(self, decoder_input, hidden, encoder_output, input_ids, max_len, teacher=None):
        """
        X : input length,
        B : batch size,
        D : hidden size,
        M : # of domains,
        J : # of slots,
        J': # of update,
        O : # of operation,
        K : max value length,
        """
        
        r"""
        Args:
            decoder_input: (B, J', D)
            hidden: (1, B, D)
            encoder_output: (B, X, D)
            input_ids: (B, X)
            max_len: int
            teacher: None or (B, J', K)

        Returns:
            all_point_outputs: (B, J', K, V)
        """
        
        device = input_ids.device
        batch_size, num_update, _ = decoder_input.shape

        mask = input_ids.eq(self.padding_idx)
        all_point_outputs = torch.zeros(batch_size, num_update, max_len, self.vocab_size).to(device)
        for j in range(num_update):
            x = decoder_input[:, j, :].unsqueeze(1) # (B, 1, D)
            for k in range(max_len):
                x = self.dropout(x)
                try:
                    _, hidden = self.decoder(x, hidden) # (1, B, D)
                except:
                    raise ValueError

                attn_vcb = torch.matmul(hidden.squeeze(0), self.embeddings.weight.transpose(0, 1)) # (B, D) x (D, V) = (B, V)
                p_vcb = F.softmax(attn_vcb, dim=-1) # (B, V)

                attn_ctx = torch.bmm(encoder_output, hidden.permute(1, 2, 0)) # (B, X, D) x (B, D, 1) = (B, X, 1)
                attn_ctx = attn_ctx.squeeze(-1).masked_fill(mask, -1e9) # (B, X)
                attn_ctx = F.softmax(attn_ctx, dim=-1) # (B, X)
                
                context = torch.bmm(attn_ctx.unsqueeze(1), encoder_output) # (B, 1, X) x (B, X, D) = (B, 1, D)
                p_gen = torch.cat([hidden.transpose(0, 1), x, context], dim=-1) # (B, 3*D)
                p_gen = self.linear_gen(p_gen) # (B, 1)
                p_gen = self.sigmoid(p_gen) # (B, 1)
                p_gen = p_gen.squeeze(-1) # (B)

                p_ctx = torch.zeros_like(p_vcb, device=device) # (B, V)
                p_ctx.scatter_add_(dim=1, index=input_ids, src=attn_ctx)

                p_svg = p_gen * p_vcb + (1 - p_gen) * p_ctx # (B, V)
                x_ids = p_svg.max(-1)[-1] # (B)

                if teacher is not None: # teacher: (B, J', K)
                    x = self.embeddings(teacher[:, j, k]).unsqueeze(1) # (B, 1, D)
                else:
                    x = self.embeddings(x_ids).unsqueeze(1) # (B, 1, D)

                all_point_outputs[:, j, k, :] = p_svg

        return all_point_outputs.contiguous() # (B, J', K, V)


class MaskedCrossEntropyLossForValue(nn.Module):
    def __init__(self, padding_idx: int = 0):
        super(MaskedCrossEntropyLossForValue, self).__init__()
        self.padding_idx = padding_idx

    def forward(self, logit, target):
        mask = target.ne(self.padding_idx)
        log_logit_flat = logit.view(-1, logit.size(-1)).log() # (B * J' * K, V)
        target_flat = target.view(-1, 1)                      # (B * J' * K, 1)
        losses_flat = -torch.gather(log_logit_flat, dim=-1, index=target_flat)
        losses = losses_flat.view(target.shape).masked_select(mask)
        loss = losses.sum() / mask.sum().float()
        return loss


class CrossEntropyLossForDomain(nn.Module):
    def forward(self, logit, target):
        log_logit = logit.log()
        target = target.float()
        losses = -(target * log_logit).sum(-1)
        loss = losses.sum() / losses.size(0)
        return loss


# class CrossEntropyLossForOperation(nn.Module):
#     def forward(self, logit, target):
#         logit = F.softmax(logit, dim=-1).log()
#         target = F.one_hot(target, logit.size(-1))
#         losses = logit * target
#         losses