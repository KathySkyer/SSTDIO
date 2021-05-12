import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, out_size):
        """初始化参数：
            vocab_size:字典的大小
            emb_size:词向量的维数
            hidden_size：隐向量的维数
            out_size:标注的种类
        """
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.bilstm = nn.LSTM(emb_size, hidden_size,
                              batch_first=True,
                              bidirectional=True)

        self.lin = nn.Linear(2*hidden_size, out_size)
        self.word_rep = WordRep(vocab_size, word_embed_dim, None, args)

    def forward(self, sents_tensor, lengths):
        emb = self.embedding(sents_tensor)  # [B, L, emb_size]

        packed = pack_padded_sequence(emb, lengths, batch_first=True)
        rnn_out, _ = self.bilstm(packed)
        # rnn_out:[B, L, hidden_size*2]
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)

        scores = self.lin(rnn_out)  # [B, L, out_size]

        return scores

    def test(self, sents_tensor, lengths, _):
        """第三个参数不会用到，加它是为了与BiLSTM_CRF保持同样的接口"""
        logits = self.forward(sents_tensor, lengths)  # [B, L, out_size]
        _, batch_tagids = torch.max(logits, dim=2)

        return batch_tagids

class WordRep(nn.Module):
    def __init__(self, vocab_size, word_embed_dim, char_size, args):
        super(WordRep, self).__init__()
        self.use_char = args.use_char
        self.use_elmo = args.use_elmo
        self.elmo_mode = args.elmo_mode
        self.elmo_mode2 = args.elmo_mode2
        self.projected = args.projected
        self.char_embed_dim = args.char_embed_dim
        self.word_embed = nn.Embedding(vocab_size, word_embed_dim)
        if self.use_elmo:
            self.elmo_weights = nn.Linear(3, 1)
            self.elmo_proj = nn.Linear(1024, word_embed_dim)
        if self.use_char:
            self.char_embed = nn.Embedding(char_size, self.char_embed_dim)
            self.char_lstm = nn.LSTM(self.char_embed_dim, self.char_embed_dim//2, num_layers=1, bidirectional=True)

    def forward(self, batch):
        sentence, _ = batch.text
        # sentence = torch.unsqueeze(sentence, -1)
        # print("sentence in wordrep")
        # print(sentence)

        # sentence = sentence.view(sentence.size()[1], -1)
        if self.use_elmo:
            elmo_tensor = batch.elmo
        else:
            elmo_tensor = None
        char_seq = None
        char_seq_len = None
        char_seq_recover = None
        words_embeds = self.word_embed(sentence)
        if self.use_elmo:
            if self.elmo_mode == 2:
                elmo_tensor = elmo_tensor[-1]
            elif self.elmo_mode == 3:
                elmo_tensor = elmo_tensor[1]
            elif self.elmo_mode == 4:
                elmo_tensor = elmo_tensor[0]
            elif self.elmo_mode == 6:
                attn_weights = F.softmax(self.elmo_weights.weight, dim=-1)
                elmo_tensor = torch.matmul(attn_weights, elmo_tensor.t())
            else:
                elmo_tensor = elmo_tensor.mean(dim=0)
            if not self.projected:
                projected = elmo_tensor
            else:
                projected = self.elmo_proj(elmo_tensor)
            # print(words_embeds.size())
            # exit(-1)
            projected = projected.view(projected.size()[0], 1, -1)
            if self.elmo_mode2 == 1:
                words_embeds = words_embeds + projected
            elif self.elmo_mode2 == 2:
                words_embeds = words_embeds
            elif self.elmo_mode2 == 3:
                words_embeds = torch.cat((words_embeds, projected), dim=-1)
            else:
                words_embeds = projected
        if self.use_char:
            char_embeds = self.char_embed(char_seq)
            pack_seq = pack_padded_sequence(char_embeds, char_seq_len, True)
            char_rnn_out, char_hidden = self.char_lstm(pack_seq)
            last_hidden = char_hidden[0].view(sentence.size()[0], 1, -1)
            # print(words_embeds)
            # print(last_hidden)
            words_embeds = torch.cat((words_embeds, last_hidden), -1)
        return words_embeds