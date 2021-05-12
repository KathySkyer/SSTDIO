import numpy as np
from torch.nn import LSTM

import networks
import torch
from torch import nn
import train

import torchtext.data as data

from MODELS.BiLSTM import BiLSTM

tag2id = {'B': 1, 'I': 2, 'O': 0}
cuda_flag = True and torch.cuda.is_available()


class SSTDIOData(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, fields, input_data, **kwargs):
        examples = []
        for e in input_data:
            examples.append(data.Example.fromlist(e, fields))
        super(SSTDIOData, self).__init__(examples, fields, **kwargs)


def numericalize(text, vocab):
    # tokens = text.split()
    ids = []
    for token in text:
        token = token.lower()
        if token in vocab:
            ids.append(vocab[token])
        else:
            ids.append(vocab['<unk>'])
            print('error:' + token)
    assert len(ids) == len(text)
    return ids


def numericalize_label(labels, vocab):
    label_tensor = []
    for i, label in enumerate(labels):
        if vocab[label] == '':
            print(i)
        label_tensor.append(vocab[label])

    return label_tensor


class NeuralTagger:  # Neural network method
    def __init__(self):
        self.word_embed_dim = 300
        self.hidden_size = 128
        self.vocab_size = 100
        self.output_size = 3
        pass

    def train_from_data(self, train_raw_data, test_raw_data, W, word2index, args):

        self.word_embed_dim = W.shape[1]
        self.hidden_size = args.n_hidden
        self.vocab_size = len(W)
        self.output_size = 3

        if args.model == 'SSTDIO':
            self.tagger = networks.SSTDIO(self.word_embed_dim, self.output_size, self.vocab_size, args)

        else:
            print("model name not found")
            exit(-1)

        W = torch.from_numpy(W)
        self.tagger.word_rep.word_embed.weight = nn.Parameter(W)

        TEXT = data.Field(sequential=True, use_vocab=False, pad_token=0, batch_first=True,
                          include_lengths=True)
        LABEL_T = data.Field(sequential=True, use_vocab=False, pad_token=0, batch_first=True)
        LABEL_O = data.Field(sequential=True, use_vocab=False, pad_token=-1, batch_first=True)
        LEFT_MASK = data.Field(sequential=True, use_vocab=False, pad_token=0, batch_first=True)
        RIGHT_MASK = data.Field(sequential=True, use_vocab=False, pad_token=0, batch_first=True)

        fields = [('text', TEXT), ('target', LABEL_T), ('label', LABEL_O), ('left_mask', LEFT_MASK), ('right_mask', RIGHT_MASK)]

        if args.use_dev:
            train_texts, train_t, train_ow, dev_texts, dev_t, dev_ow = self.split_dev(*train_raw_data)
            dev_data = [
                [numericalize(text, word2index), numericalize_label(target, tag2id),
                 numericalize_label(label, tag2id), *self.generate_mask(target)]
                for text, target, label in zip(dev_texts, dev_t, dev_ow)]
            dev_dataset = SSTDIOData(fields, dev_data)
        train_data = [
            [numericalize(text, word2index), numericalize_label(target, tag2id),
             numericalize_label(label, tag2id), *self.generate_mask(target)]
            for text, target, label in zip(train_texts, train_t, train_ow)]
        test_data = [
            [numericalize(text, word2index), numericalize_label(target, tag2id),
             numericalize_label(label, tag2id), *self.generate_mask(target)]
            for text, target, label in zip(*test_raw_data)]
        train_dataset = SSTDIOData(fields, train_data)
        test_dataset = SSTDIOData(fields, test_data)

        device = torch.device("cuda" if torch.cuda.is_available() and cuda_flag else "cpu")
        n_gpu = torch.cuda.device_count()
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)

        train_iter = data.Iterator(train_dataset, batch_size=args.batch_size, sort_within_batch=True, repeat=False,
                                   device=device if torch.cuda.is_available() else -1)
        if args.use_dev:
            dev_iter = data.Iterator(dev_dataset, batch_size=args.eval_bs, shuffle=False, sort_within_batch=True,
                                  repeat=False,
                                  device=device if torch.cuda.is_available() else -1)
        else:
            dev_iter = None
        test_iter = data.Iterator(test_dataset, batch_size=args.eval_bs, shuffle=False, sort_within_batch=True,
                                  repeat=False,
                                  device=device if torch.cuda.is_available() else -1)
        train.train(self.tagger, train_iter, dev_iter, test_iter, args=args)
        pass

    def split_dev(self, train_texts, train_t, train_ow):
        instances_index = []
        curr_s = ""
        curr_i = -1
        for i, s in enumerate(train_texts):
            s = ' '.join(s)

            if s == curr_s:
                instances_index[curr_i].append(i)
            else:
                curr_s = s
                instances_index.append([i])
                curr_i += 1
        print(curr_i)
        print(len(instances_index))
        assert curr_i+1 == len(instances_index)
        length = len(instances_index)
        np.random.seed(1024)
        index_list = np.random.permutation(length).tolist()
        train_index = [instances_index[i] for i in index_list[0:length-length//5]]
        dev_index = [instances_index[i] for i in index_list[length-length//5:]]
        train_i_index = [i for l in train_index for i in l]
        dev_i_index = [i for l in dev_index for i in l]
        dev_texts, dev_t, dev_ow = ([train_texts[i] for i in dev_i_index], [train_t[i] for i in dev_i_index],
                                    [train_ow[i] for i in dev_i_index])
        train_texts, train_t, train_ow = ([train_texts[i] for i in train_i_index], [train_t[i] for i in train_i_index],
                                          [train_ow[i] for i in train_i_index])
        return train_texts, train_t, train_ow, dev_texts, dev_t, dev_ow


    def generate_mask(self, target):
        labels = numericalize_label(target, tag2id)
        index = np.nonzero(labels)
        if index[0] != []:
            start = index[0][0]
            end = index[0][-1]
            left_mask = np.asarray([1 for _ in range(len(target))])
            right_mask = np.asarray([1 for _ in range(len(target))])
            left_mask[end + 1:] = 0
            right_mask[:start] = 0
        else:
            left_mask = np.asarray([1 for _ in range(len(target))])
            right_mask = np.asarray([0 for _ in range(len(target))])
        return left_mask, right_mask


if __name__ == '__main__':
    model = PTBTagger()
    sentence = ["i recommend this place to everyone .".split()]
    target = "0 0 0 1 0 0 0".split()
    target = [[int(i) for i in target]]
    ow = model.predict(sentence, target)
    print(ow)
