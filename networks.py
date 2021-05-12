import torch
from sklearn_crfsuite import CRF
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math

class WordRep(nn.Module):
    def __init__(self, vocab_size, word_embed_dim):
        super(WordRep, self).__init__()
        self.word_embed = nn.Embedding(vocab_size, word_embed_dim)

    def forward(self, batch):
        sentence, _ = batch.text
        words_embeds = self.word_embed(sentence)
        return words_embeds

class SSTDIO(nn.Module):
    def __init__(self, word_embed_dim, output_size, vocab_size, args=None):
        super(SSTDIO, self).__init__()
        self.deeplayer = args.num_layers
        self.l1 = args.l1
        self.input_size = word_embed_dim
        self.dropout = args.dropout
        self.hidden_size = args.n_hidden
        self.output_size = output_size
        self.max_length = args.batch_size
        self.word_rep = WordRep(vocab_size, word_embed_dim)
        self.IO = IO(word_embed_dim, self.l1, self.hidden_size,self.deeplayer)
        self.CNN = nn.Conv1d(self.input_size, self.hidden_size, 5, padding=2)
        self.CNN_LSTM = nn.LSTM(self.hidden_size,self.hidden_size,2, bidirectional=True, batch_first=True)

        self.atten = Attention(self.hidden_size * 2, n_head=8, score_function='bi_linear', dropout=self.dropout)

        # self.convs = nn.ModuleList([nn.Conv2d(chanel_num, filter_num, (size, embedding_dimension)) for size in filter_sizes])
        self.fc = nn.Linear(self.hidden_size * 2, self.output_size)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, batch):
        target = batch.target
        sentence = self.word_rep(batch)
        # target + sentence(DIO)
        sentences_encoded, target = self.IO(sentence, batch.left_mask, batch.right_mask, target, self.deeplayer)
        sentences_tarencoded = F.relu(sentences_encoded + target)
        # sentences_encoded = sentences_tarencoded + sentences_encoded

        # intensity 生成
        sentences_temp = self.CNN(self.dropout(sentence).transpose(1, 2))
        sentence_tem = F.relu(sentences_temp)
        sentence_intensity, _ = self.CNN_LSTM(sentence_tem.transpose(1, 2))

        # aten_weight, _ = self.atten(target, sentences_encoded)
        # encoded = torch.mul(aten_weight,sentences_encoded)

        # encoded = torch.mul()
        # encoded = torch.mul(sentence_intensity,sentences_tarencoded)


        encoded = F.relu(sentence_intensity + sentences_encoded)

         #CNN + SSDIO
        # cnn_sentences = self.CNN(self.dropout(sentence).transpose(1, 2))
        # encoded = torch.mul(target,cnn_sentences)
        # encoded = F.relu(torch.mul(encoded,sentence_intensity))




        decodedP = self.fc(encoded)
        outputP = F.log_softmax(decodedP, dim=-1)
        return outputP


class IO(nn.Module):
    def __init__(self, word_embed_dim, l1, n_hidden, num_layers):
        super(IO, self).__init__()
        self.input_size = word_embed_dim
        self.hidden_size = n_hidden
        self.num_layers = num_layers
        self.rnn_L = nn.LSTM(self.input_size, self.hidden_size, num_layers=l1, bidirectional=True, batch_first=True)
        self.rnn_R = nn.LSTM(self.input_size, self.hidden_size, num_layers=l1, bidirectional=True, batch_first=True)

    def forward(self, sentence,left_mask,right_mask,target,num_layers):
        target_mask = target != 0

        left_context = sentence * left_mask.unsqueeze(-1).float().expand_as(sentence)
        right_context = sentence * right_mask.unsqueeze(-1).float().expand_as(sentence)
        left_encoded, right_encoded = left_context,right_context

        left_encoded, _ = self.rnn_L(left_encoded)
        right_encoded, _ = self.rnn_R(right_encoded)

        left_encoded = left_encoded * left_mask.unsqueeze(-1).float().expand_as(left_encoded)
        right_encoded = right_encoded * right_mask.unsqueeze(-1).float().expand_as(right_encoded)


        encoded = left_encoded + right_encoded
        target_average_mask = 1 - 1/2*target_mask.unsqueeze(-1).float().expand_as(encoded)
        encoded = encoded * target_average_mask
        return encoded,target_average_mask


class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0):
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:  # dot_product / scaled_dot_product
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            # kq = torch.unsqueeze(kx, dim=1) + torch.unsqueeze(qx, dim=2)
            score = torch.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=-1)
        output = torch.bmm(score, kx)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (?, q_len, n_head*hidden_dim)
        output = self.proj(output)
        output = self.dropout(output)
        return output, score


