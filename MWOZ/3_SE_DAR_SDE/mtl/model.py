from transformers import AdamW, BertTokenizer, BertModel
import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.normalization import LayerNorm
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F
from tqdm import tqdm
import copy
from transformer import TransformerEncoder
import torchcrf


def init_params(model):
    for name, param in model.named_parameters():
        if param.data.dim() > 1:
            xavier_uniform_(param.data)
        else:
            pass


def universal_sentence_embedding(sentences, mask, sqrt=True):
    sentence_sums = torch.bmm(
        sentences.permute(0, 2, 1), mask.float().unsqueeze(-1)
    ).squeeze(-1)
    divisor = (mask.sum(dim=1).view(-1, 1).float())
    if sqrt:
        divisor = divisor.sqrt()
    sentence_sums /= divisor
    return sentence_sums


class Pretrained_HiTrans(nn.Module):
    def __init__(self, args, vocab_size):
        super().__init__()
        self.drop_out1 = nn.Dropout(args.dropout)
        self.drop_out2 = nn.Dropout(args.dropout)
        self.private = BERTBackbone(bert_name=args.bert_name, cache_dir=args.cache_dir)
        d_model = self.private.d_model

        #self.gru = nn.GRU(d_model, d_model, num_layers=1, bidirectional=False, batch_first=True)

        self.class_num = 2
        self.encoder = TransformerEncoder(d_model, d_model*2, 8, 2, 0.1)
        #self.act_classifier = MLP(d_model, self.class_num, d_model//2)
        self.act_classifier = nn.Linear(d_model, self.class_num)
        self.sat_classifier = nn.Linear(d_model, 2)
        #self.sat_classifier = MLP(d_model, 2, d_model//2)
        
        init_params(self.act_classifier)
        init_params(self.encoder)
        init_params(self.sat_classifier)

    def forward(self, input_ids, act_seq=None, sat=None, **kwargs):
        #self.gru.flatten_parameters()

        batch_size, dialog_len, utt_len = input_ids.size()

        input_ids = input_ids.view(-1, utt_len)
        attention_mask = act_seq.ne(-1).detach()

        private_out = self.private(input_ids=input_ids, **kwargs)
        private_out = private_out.view(batch_size, dialog_len, -1) 
        H = self.encoder(private_out, attention_mask)
        hidden = universal_sentence_embedding(H, attention_mask)
        private_out = self.drop_out1(private_out)

        #_, hidden = self.gru(H)
        #hidden = hidden.squeeze(0)
        
        hidden = self.drop_out2(hidden)

        act_res = self.act_classifier(private_out)
        sat_res = self.sat_classifier(hidden)

        if self.training:
            act_loss = F.cross_entropy(act_res.view(-1, self.class_num), act_seq.view(-1), ignore_index=-1)
            sat_loss = F.cross_entropy(sat_res, sat)
            return act_res, sat_res, act_loss, sat_loss

        return act_res, sat_res


class FocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """
 
    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average
 
        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')
 
        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')
 
    def forward(self, input, target):
        logit = F.softmax(input, dim=1)
 
        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)
 
        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)
 
        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)
 
        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()
 
        gamma = self.gamma
 
        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt
 
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class BERTBackbone(nn.Module):
    def __init__(self, **config):
        super().__init__()
        bert_name = config.get('bert_name', 'bert-base-uncased')
        cache_dir = config.get('cache_dir')
        self.bert = BertModel.from_pretrained(bert_name, cache_dir=cache_dir)
        self.d_model = 768*2

    def forward(self, input_ids, **kwargs):
        attention_mask = input_ids.ne(0).detach()
        outputs = self.bert(input_ids, attention_mask)
        h = universal_sentence_embedding(outputs[0], attention_mask)
        cls = outputs[1]

        out = torch.cat([cls, h], dim=-1)
        return out


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, din):
        dout = F.relu(self.fc1(din))
        dout = F.relu(self.fc2(dout))
        return dout


class SE_DAR_SDE(nn.Module):
    def __init__(self, args, vocab_size, class_num, pretrained_private=None, pretrained_encoder=None):
        super().__init__()
        self.drop_out = nn.Dropout(args.dropout)
        if pretrained_private is not None:
            self.private = pretrained_private
        else:
            self.private = BERTBackbone(bert_name=args.bert_name, cache_dir=args.cache_dir)
        d_model = self.private.d_model

        self.class_num = class_num
        if pretrained_encoder is not None:
            self.encoder = pretrained_encoder
        else:
            self.encoder = TransformerEncoder(d_model, d_model*2, 8, 2, 0.1)

        self.content_gru = nn.GRU(d_model, d_model, num_layers=2, bidirectional=False, batch_first=True)
        self.act_gru = nn.GRU(class_num, d_model, num_layers=2, bidirectional=False, batch_first=True)

        self.act_classifier = MLP(d_model, class_num, d_model//2)
        self.sat_diff_classifier = nn.Linear(d_model, 5)
        self.sat_classifier = nn.Linear(d_model, 3)
        self.crf = torchcrf.CRF(class_num, batch_first=True)

        self.U_c = nn.Linear(d_model, d_model)
        self.w_c = nn.Linear(d_model, 1, bias=False)
        self.U_a = nn.Linear(d_model, d_model)
        self.w_a = nn.Linear(d_model, 1, bias=False)

        self.w = nn.Linear(d_model*2, 1)
        
        init_params(self.act_classifier)
        init_params(self.encoder)
        init_params(self.sat_diff_classifier)
        init_params(self.sat_classifier)
        init_params(self.w)
        init_params(self.U_c)
        init_params(self.w_c)
        init_params(self.U_a)
        init_params(self.w_a)

    def forward(self, input_ids, act_seq=None, sat_diff=None, sat=None, **kwargs):
        self.content_gru.flatten_parameters()
        self.act_gru.flatten_parameters()

        batch_size, dialog_len, utt_len = input_ids.size()

        input_ids = input_ids.view(-1, utt_len)
        attention_mask = act_seq.ne(-1).detach()

        private_out = self.private(input_ids=input_ids, **kwargs)
        private_out = private_out.view(batch_size, dialog_len, -1) 
        H = self.encoder(private_out, attention_mask)
        H = self.drop_out(H)

        act_res = self.act_classifier(H)

        H, _ = self.content_gru(H)
        att_c = self.w_c(torch.tanh(self.U_c(H))).squeeze(-1)
        att_c = F.softmax(att_c.masked_fill(mask=~attention_mask, value=-np.inf), dim=1)
        content_hidden = torch.bmm(H.permute(0,2,1), att_c.unsqueeze(-1)).squeeze(-1)

        H, _ = self.act_gru(act_res)
        att_a = self.w_a(torch.tanh(self.U_a(H))).squeeze(-1)
        att_a = F.softmax(att_a.masked_fill(mask=~attention_mask, value=-np.inf), dim=1)
        act_hidden = torch.bmm(H.permute(0,2,1), att_a.unsqueeze(-1)).squeeze(-1)

        hidden = torch.cat([content_hidden, act_hidden], dim=-1)
        pointer = torch.sigmoid(self.w(hidden))
        hidden = content_hidden * pointer + act_hidden * (1-pointer)
        
        sat_res = self.sat_classifier(hidden)
        sat_diff_res = self.sat_diff_classifier(hidden)

        if self.training:
            #act_loss = F.cross_entropy(act_res.view(-1, self.class_num), act_seq.view(-1), ignore_index=-1)
            act_loss = -1*self.crf(act_res, act_seq, mask=attention_mask)
                      
            weight_class2 = torch.FloatTensor([9.475324675324675,3.2835283528352837,1.0,2.881516587677725,8.706443914081145]).cuda()
            sat_diff_loss = F.cross_entropy(sat_diff_res, sat_diff, weight=weight_class2)
            
            weight_class1 = torch.FloatTensor([1.4405263157894737, 1.0, 1.248631386861314]).cuda()
            sat_loss = F.cross_entropy(sat_res, sat, weight=weight_class1)

            return act_res, sat_diff_res, sat_res, act_loss, sat_diff_loss,sat_loss       

        return act_res, sat_diff_res, sat_res, pointer, att_a, att_c

