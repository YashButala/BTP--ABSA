import sys
import os
import numpy as np
import random

from collections import OrderedDict
import pickle
import datetime
import json
from tqdm import tqdm
from recordclass import recordclass
import math
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.backends.cudnn.deterministic = True

import nltk
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import SpaceTokenizer


def custom_print(*msg):
    for i in range(0, len(msg)):
        if i == len(msg) - 1:
            print(msg[i])
            logger.write(str(msg[i]) + '\n')
        else:
            print(msg[i], ' ', end='')
            logger.write(str(msg[i]))


def load_word_embedding(embed_file, vocab):
    custom_print('vocab length:', len(vocab))
    custom_print(embed_file)
    embed_vocab = OrderedDict()
    embed_matrix = list()

    embed_vocab['<PAD>'] = 0
    embed_matrix.append(np.zeros(word_embed_dim, dtype=np.float32))

    embed_vocab['<UNK>'] = 1
    embed_matrix.append(np.random.uniform(-0.25, 0.25, word_embed_dim))

    word_idx = 2
    with open(embed_file, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < word_embed_dim + 1:
                continue
            word = parts[0]
            if word in vocab and word not in embed_vocab and vocab[word] >= word_min_freq:
                vec = [np.float32(val) for val in parts[1:]]
                embed_matrix.append(vec)
                embed_vocab[word] = word_idx
                word_idx += 1

    for word in vocab:
        if word not in embed_vocab and vocab[word] >= word_min_freq:
            embed_matrix.append(np.random.uniform(-0.25, 0.25, word_embed_dim))
            embed_vocab[word] = word_idx
            word_idx += 1

    custom_print('embed dictionary length:', len(embed_vocab))
    return embed_vocab, np.array(embed_matrix, dtype=np.float32)


def build_vocab(data, save_vocab, embedding_file):
    vocab = OrderedDict()
    char_v = OrderedDict()
    char_v['<PAD>'] = 0
    char_v['<UNK>'] = 1
    char_idx = 2
    for d in data:
        for word in d.SrcWords:
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1

            for c in word:
                if c not in char_v:
                    char_v[c] = char_idx
                    char_idx += 1

    word_v, embed_matrix = load_word_embedding(embedding_file, vocab)
    output = open(save_vocab, 'wb')
    pickle.dump([word_v, char_v, pos_vocab], output)
    output.close()
    return word_v, char_v, embed_matrix


def build_tags(file1, file2, file3):
    tk = SpaceTokenizer()
    f1 = open(file1, "r")
    f2 = open(file2, "r")
    f3 = open(file3, "r")
    pos_vocab = OrderedDict()
    pos_vocab['<PAD>'] = 0
    pos_vocab['<UNK>'] = 1
    k = 2
    for line in f1:
        text = tk.tokenize(line)
        tags = nltk.pos_tag(text)
        for t in tags:
            if t[1] not in pos_vocab:
                pos_vocab[t[1]] = k
                k += 1
    for line in f2:
        text = tk.tokenize(line)
        tags = nltk.pos_tag(text)
        for t in tags:
            if t[1] not in pos_vocab:
                pos_vocab[t[1]] = k
                k += 1
    for line in f3:
        text = tk.tokenize(line)
        tags = nltk.pos_tag(text)
        for t in tags:
            if t[1] not in pos_vocab:
                pos_vocab[t[1]] = k
                k += 1
    return pos_vocab


def load_vocab(vocab_file):
    with open(vocab_file, 'rb') as f:
        embed_vocab, char_vocab, pos_vocab = pickle.load(f)
    return embed_vocab, char_vocab, pos_vocab


def get_data(src_lines, trg_lines, datatype):
    samples = []
    uid = 1
    for i in range(0, len(src_lines)):
        src_line = src_lines[i].strip()
        trg_line = trg_lines[i].strip()
        src_words = src_line.split(' ')

        trg_rels = []
        trg_pointers = []
        parts = trg_line.split('|')
        if datatype == 1:
            random.shuffle(parts)
        for part in parts:
            elements = part.strip().split()
            trg_rels.append(relnameToIdx[elements[4]])
            trg_pointers.append((int(elements[0]), int(elements[1]), int(elements[2]), int(elements[3])))

        if datatype == 1 and (len(src_words) > max_src_len or len(trg_rels) > max_trg_len):
            continue

        sample = Sample(Id=uid, SrcLen=len(src_words), SrcWords=src_words, TrgLen=len(trg_rels), TrgRels=trg_rels,
                        TrgPointers=trg_pointers)
        samples.append(sample)
        uid += 1

        if use_data_aug and datatype == 1 and len(parts) > 1:
            random.shuffle(parts)
            for part in parts:
                elements = part.strip().split()
                trg_rels.append(relnameToIdx[elements[4]])
                trg_pointers.append((int(elements[0]), int(elements[1]), int(elements[2]), int(elements[3])))
            sample = Sample(Id=uid, SrcLen=len(src_words), SrcWords=src_words, TrgLen=len(trg_rels),
                            TrgRels=trg_rels, TrgPointers=trg_pointers)
            samples.append(sample)
            uid += 1

    return samples


def read_data(src_file, trg_file,  datatype):
    reader = open(src_file)
    src_lines = reader.readlines()
    reader.close()

    reader = open(trg_file)
    trg_lines = reader.readlines()
    reader.close()

    data = get_data(src_lines, trg_lines,  datatype)
    return data


def get_relations(file_name):
    nameToIdx = OrderedDict()
    idxToName = OrderedDict()
    reader = open(file_name)
    lines = reader.readlines()
    reader.close()
    nameToIdx['<PAD>'] = 0
    idxToName[0] = '<PAD>'
    # nameToIdx['<SOS>'] = 1
    # idxToName[1] = '<SOS>'
    nameToIdx['None'] = 1
    idxToName[1] = 'None'
    idx = 2
    for line in lines:
        nameToIdx[line.strip()] = idx
        idxToName[idx] = line.strip()
        idx += 1
    return nameToIdx, idxToName


def get_answer_pointers(arg1start_preds, arg1end_preds, arg2start_preds, arg2end_preds, sent_len):
    arg1_prob = -1.0
    arg1start = -1
    arg1end = -1
    max_ent_len = 5
    for i in range(0, sent_len):
        for j in range(i, min(sent_len, i + max_ent_len)):
            if arg1start_preds[i] * arg1end_preds[j] > arg1_prob:
                arg1_prob = arg1start_preds[i] * arg1end_preds[j]
                arg1start = i
                arg1end = j

    arg2_prob = -1.0
    arg2start = -1
    arg2end = -1
    for i in range(0, arg1start):
        for j in range(i, min(arg1start, i + max_ent_len)):
            if arg2start_preds[i] * arg2end_preds[j] > arg2_prob:
                arg2_prob = arg2start_preds[i] * arg2end_preds[j]
                arg2start = i
                arg2end = j
    for i in range(arg1end + 1, sent_len):
        for j in range(i, min(sent_len, i + max_ent_len)):
            if arg2start_preds[i] * arg2end_preds[j] > arg2_prob:
                arg2_prob = arg2start_preds[i] * arg2end_preds[j]
                arg2start = i
                arg2end = j

    arg2_prob1 = -1.0
    arg2start1 = -1
    arg2end1 = -1
    for i in range(0, sent_len):
        for j in range(i, min(sent_len, i + max_ent_len)):
            if arg2start_preds[i] * arg2end_preds[j] > arg2_prob1:
                arg2_prob1 = arg2start_preds[i] * arg2end_preds[j]
                arg2start1 = i
                arg2end1 = j

    arg1_prob1 = -1.0
    arg1start1 = -1
    arg1end1 = -1
    for i in range(0, arg2start1):
        for j in range(i, min(arg2start1, i + max_ent_len)):
            if arg1start_preds[i] * arg1end_preds[j] > arg1_prob1:
                arg1_prob1 = arg1start_preds[i] * arg1end_preds[j]
                arg1start1 = i
                arg1end1 = j
    for i in range(arg2end1 + 1, sent_len):
        for j in range(i, min(sent_len, i + max_ent_len)):
            if arg1start_preds[i] * arg1end_preds[j] > arg1_prob1:
                arg1_prob1 = arg1start_preds[i] * arg1end_preds[j]
                arg1start1 = i
                arg1end1 = j
    if arg1_prob * arg2_prob > arg1_prob1 * arg2_prob1:
        return arg1start, arg1end, arg2start, arg2end
    else:
        return arg1start1, arg1end1, arg2start1, arg2end1


def is_full_match(triplet, triplets):
    for t in triplets:
        if t[0] == triplet[0] and t[1] == triplet[1] and t[2] == triplet[2]:
            return True
    return False


def get_gt_triples(src_words, rels, pointers):
    triples = []
    i = 0
    for r in rels:
        arg1 = ' '.join(src_words[pointers[i][0]:pointers[i][1] + 1])
        arg2 = ' '.join(src_words[pointers[i][2]:pointers[i][3] + 1])
        triplet = (arg1.strip(), arg2.strip(), relIdxToName[r])
        if not is_full_match(triplet, triples):
            triples.append(triplet)
        i += 1
    return triples


def get_pred_triples(rel, arg1s, arg1e, arg2s, arg2e, src_words):
    triples = []
    all_triples = []
    for i in range(0, len(rel)):
        r = np.argmax(rel[i][1:]) + 1
        if r == relnameToIdx['None']:
            break
        s1, e1, s2, e2 = get_answer_pointers(arg1s[i], arg1e[i], arg2s[i], arg2e[i], len(src_words))
        arg1 = ' '.join(src_words[s1: e1 + 1])
        arg2 = ' '.join(src_words[s2: e2 + 1])
        arg1 = arg1.strip()
        arg2 = arg2.strip()
        if arg1 == arg2:
            continue
        triplet = (arg1, arg2, relIdxToName[r])
        all_triples.append(triplet)
        if not is_full_match(triplet, triples):
            triples.append(triplet)
    return triples, all_triples


def get_F1(data, preds):
    gt_pos = 0
    pred_pos = 0
    total_pred_pos = 0
    correct_pos = 0
    for i in range(0, len(data)):
        gt_triples = get_gt_triples(data[i].SrcWords, data[i].TrgRels, data[i].TrgPointers)

        pred_triples, all_pred_triples = get_pred_triples(preds[0][i], preds[1][i], preds[2][i], preds[3][i],
                                                          preds[4][i], data[i].SrcWords)
        total_pred_pos += len(all_pred_triples)
        gt_pos += len(gt_triples)
        pred_pos += len(pred_triples)
        for gt_triple in gt_triples:
            if is_full_match(gt_triple, pred_triples):
                correct_pos += 1
    print(total_pred_pos)
    return pred_pos, gt_pos, correct_pos


def write_test_res(src, trg, data, preds, outfile):
    reader = open(src)
    src_lines = reader.readlines()
    writer = open(outfile, 'w')
    for i in range(0, len(data)):
        writer.write(src_lines[i])
        writer.write('Expected: '+trg[i])
        pred_triples, _ = get_pred_triples(preds[0][i], preds[1][i], preds[2][i], preds[3][i], preds[4][i],
                                        data[i].SrcWords)
        pred_triples_str = []
        for pt in pred_triples:
            str_tmp = pt[0] + ' ; ' + pt[1] + ' ; ' + pt[2]
            if str_tmp not in pred_triples_str:
                pred_triples_str.append(str_tmp)
        writer.write('Predicted: ' + ' | '.join(pred_triples_str) + '\n'+'\n')
    writer.close()
    reader.close()


def shuffle_data(data):
    custom_print(len(data))
    data.sort(key=lambda x: x.SrcLen)
    num_batch = int(len(data) / batch_size)
    rand_idx = random.sample(range(num_batch), num_batch)
    new_data = []
    for idx in rand_idx:
        new_data += data[batch_size * idx: batch_size * (idx + 1)]
    if len(new_data) < len(data):
        new_data += data[num_batch * batch_size:]
    return new_data


def get_max_len(sample_batch):
    src_max_len = len(sample_batch[0].SrcWords)
    for idx in range(1, len(sample_batch)):
        if len(sample_batch[idx].SrcWords) > src_max_len:
            src_max_len = len(sample_batch[idx].SrcWords)

    trg_max_len = len(sample_batch[0].TrgRels)
    for idx in range(1, len(sample_batch)):
        if len(sample_batch[idx].TrgRels) > trg_max_len:
            trg_max_len = len(sample_batch[idx].TrgRels)

    return src_max_len, trg_max_len


def get_words_index_seq(words, max_len):
    seq = list()
    for word in words:
        if word in word_vocab:
            seq.append(word_vocab[word])
        else:
            seq.append(word_vocab['<UNK>'])
    pad_len = max_len - len(words)
    for i in range(0, pad_len):
        seq.append(word_vocab['<PAD>'])
    return seq


def get_pos_index_seq(words, max_len):
    seq = list()
    tk = SpaceTokenizer()
    sent = " ".join(words)
    text = tk.tokenize(sent)
    tags = nltk.pos_tag(text)
    for t in tags:
        t1 = t[1]
        if t1 in pos_vocab:
            seq.append(pos_vocab[t1])
        else:
            seq.append(pos_vocab['<UNK>'])
    pad_len = max_len - len(seq)
    for i in range(0, pad_len):
        seq.append(pos_vocab['<PAD>'])
    return seq


def get_char_seq(words, max_len):
    char_seq = list()
    for i in range(0, conv_filter_size - 1):
        char_seq.append(char_vocab['<PAD>'])
    for word in words:
        for c in word[0:min(len(word), max_word_len)]:
            if c in char_vocab:
                char_seq.append(char_vocab[c])
            else:
                char_seq.append(char_vocab['<UNK>'])
        pad_len = max_word_len - len(word)
        for i in range(0, pad_len):
            char_seq.append(char_vocab['<PAD>'])
        for i in range(0, conv_filter_size - 1):
            char_seq.append(char_vocab['<PAD>'])

    pad_len = max_len - len(words)
    for i in range(0, pad_len):
        for i in range(0, max_word_len + conv_filter_size - 1):
            char_seq.append(char_vocab['<PAD>'])
    return char_seq


def get_relation_index_seq(rel_ids, max_len):
    seq = list()
    for r in rel_ids:
        seq.append(r)
    seq.append(relnameToIdx['None'])
    pad_len = max_len + 1 - len(seq)
    for i in range(0, pad_len):
        seq.append(relnameToIdx['<PAD>'])
    return seq


def get_padded_pointers(pointers, pidx, max_len):
    idx_list = []
    for p in pointers:
        idx_list.append(p[pidx])
    pad_len = max_len + 1 - len(pointers)
    for i in range(0, pad_len):
        idx_list.append(-1)
    return idx_list


def get_pointer_location(pointers, pidx, src_max_len, trg_max_len):
    loc_seq = []
    for p in pointers:
        cur_seq = [0 for i in range(src_max_len)]
        cur_seq[p[pidx]] = 1
        loc_seq.append(cur_seq)
    pad_len = trg_max_len + 1 - len(pointers)
    for i in range(pad_len):
        cur_seq = [0 for i in range(src_max_len)]
        loc_seq.append(cur_seq)
    return loc_seq


def get_padded_mask(cur_len, max_len):
    mask_seq = list()
    for i in range(0, cur_len):
        mask_seq.append(0)
    pad_len = max_len - cur_len
    for i in range(0, pad_len):
        mask_seq.append(1)
    return mask_seq


def get_target_vec(pointers, rels, src_max_len):
    vec = [0 for i in range(src_max_len + len(relnameToIdx))]
    for i in range(len(pointers)):
        p = pointers[i]
        vec[p[0]] += 1
        vec[p[1]] += 1
        vec[p[2]] += 1
        vec[p[3]] += 1
        vec[src_max_len + rels[i]] += 1
    return vec


def get_batch_data(cur_samples, is_training=False):
    """
    Returns the training samples and labels as numpy array
    """
    batch_src_max_len, batch_trg_max_len = get_max_len(cur_samples)
    batch_trg_max_len += 1
    src_words_list = list()
    src_words_mask_list = list()
    src_char_seq = list()
    decoder_input_list = list()
    src_pos_seq = list()
    arg1sweights = []
    arg1eweights = []
    arg2sweights = []
    arg2eweights = []

    rel_seq = list()
    arg1_start_seq = list()
    arg1_end_seq = list()
    arg2_start_seq = list()
    arg2_end_seq = list()
    target_vec_seq = []
    target_vec_mask_seq = []

    for sample in cur_samples:
        src_words_list.append(get_words_index_seq(sample.SrcWords, batch_src_max_len))
        src_words_mask_list.append(get_padded_mask(sample.SrcLen, batch_src_max_len))
        src_char_seq.append(get_char_seq(sample.SrcWords, batch_src_max_len))
        src_pos_seq.append(get_pos_index_seq(sample.SrcWords, batch_src_max_len))

        if is_training:
            arg1_start_seq.append(get_padded_pointers(sample.TrgPointers, 0, batch_trg_max_len))
            arg1_end_seq.append(get_padded_pointers(sample.TrgPointers, 1, batch_trg_max_len))
            arg2_start_seq.append(get_padded_pointers(sample.TrgPointers, 2, batch_trg_max_len))
            arg2_end_seq.append(get_padded_pointers(sample.TrgPointers, 3, batch_trg_max_len))
            arg1sweights.append(get_pointer_location(sample.TrgPointers, 0, batch_src_max_len, batch_trg_max_len))
            arg1eweights.append(get_pointer_location(sample.TrgPointers, 1, batch_src_max_len, batch_trg_max_len))
            arg2sweights.append(get_pointer_location(sample.TrgPointers, 2, batch_src_max_len, batch_trg_max_len))
            arg2eweights.append(get_pointer_location(sample.TrgPointers, 3, batch_src_max_len, batch_trg_max_len))
            rel_seq.append(get_relation_index_seq(sample.TrgRels, batch_trg_max_len))
            decoder_input_list.append(get_relation_index_seq(sample.TrgRels, batch_trg_max_len))
            target_vec_seq.append(get_target_vec(sample.TrgPointers, sample.TrgRels, batch_src_max_len))
            target_vec_mask_seq.append([0 for i in range(len(sample.TrgRels))] +
                                       [1 for i in range(batch_trg_max_len + 1 - len(sample.TrgRels))])
        else:
            decoder_input_list.append(get_relation_index_seq([], 1))

    return {'src_words': np.array(src_words_list, dtype=np.float32),
            'src_words_mask': np.array(src_words_mask_list),
            'src_chars': np.array(src_char_seq),
            'src_pos_tags': np.array(src_pos_seq),
            'decoder_input': np.array(decoder_input_list),
            'arg1sweights': np.array(arg1sweights),
            'arg1eweights': np.array(arg1eweights),
            'arg2sweights': np.array(arg2sweights),
            'arg2eweights': np.array(arg2eweights),
            'rel': np.array(rel_seq),
            'arg1_start': np.array(arg1_start_seq),
            'arg1_end': np.array(arg1_end_seq),
            'arg2_start': np.array(arg2_start_seq),
            'arg2_end': np.array(arg2_end_seq),
            'target_vec': np.array(target_vec_seq),
            'target_vec_mask': np.array(target_vec_mask_seq)}


class WordEmbeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, drop_out_rate):
        super(WordEmbeddings, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if job_mode == 'train':
            self.embeddings.weight.data.copy_(torch.from_numpy(word_embed_matrix))
        self.dropout = nn.Dropout(drop_out_rate)

    def forward(self, words_seq):
        word_embeds = self.embeddings(words_seq)
        word_embeds = self.dropout(word_embeds)
        return word_embeds

    def weight(self):
        return self.embeddings.weight


class CharEmbeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, drop_out_rate):
        super(CharEmbeddings, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1d = nn.Conv1d(char_embed_dim, char_feature_size, conv_filter_size)
        self.max_pool = nn.MaxPool1d(max_word_len + conv_filter_size - 1, max_word_len + conv_filter_size - 1)
        self.dropout = nn.Dropout(drop_out_rate)

    def forward(self, char_seq):
        char_embeds = self.embeddings(char_seq)
        char_embeds = self.dropout(char_embeds)
        char_embeds = char_embeds.permute(0, 2, 1)
        char_feature = torch.tanh(self.max_pool(self.conv1d(char_embeds)))
        char_feature = char_feature.permute(0, 2, 1)
        return char_feature


class POSEmbeddings(nn.Module):
    def __init__(self, tag_len, tag_dim, drop_out_rate):
        super(POSEmbeddings, self).__init__()
        self.embeddings = nn.Embedding(tag_len, tag_dim, padding_idx=0)
        self.dropout = nn.Dropout(drop_out_rate)

    def forward(self, pos_seq):
        pos_embeds = self.embeddings(pos_seq)
        pos_embeds = self.dropout(pos_embeds)
        return pos_embeds


class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.linear_ctx = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.linear_query = nn.Linear(self.input_dim, self.input_dim, bias=True)
        self.v = nn.Linear(self.input_dim, 1)

    def forward(self, s_prev, enc_hs, src_mask):
        uh = self.linear_ctx(enc_hs)
        wq = self.linear_query(s_prev)
        wquh = torch.tanh(wq + uh)
        attn_weights = self.v(wquh).squeeze()
        attn_weights.data.masked_fill_(src_mask.data, -float('inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        ctx = torch.bmm(attn_weights.unsqueeze(1), enc_hs).squeeze()
        return ctx, attn_weights


class Sentiment_Attention(nn.Module):
    def __init__(self, enc_hid_dim, arg_dim):
        super(Sentiment_Attention, self).__init__()
        self.w1 = nn.Linear(enc_hid_dim + arg_dim, 1)
        self.w2 = nn.Linear(enc_hid_dim + arg_dim, 1)

    def forward(self, arg1, arg2, enc_hs, src_mask):
        ctx_arg1 = torch.cat((enc_hs, arg1.unsqueeze(1).repeat(1, enc_hs.size()[1], 1)), -1)
        ctx_arg1_att = self.w1(ctx_arg1).squeeze()
        ctx_arg1_att.data.masked_fill_(src_mask.data, -float('inf'))
        ctx_arg1_att = F.softmax(ctx_arg1_att, dim=-1)
        ctx1 = torch.bmm(ctx_arg1_att.unsqueeze(1), enc_hs).squeeze()

        ctx_arg2 = torch.cat((enc_hs, arg2.unsqueeze(1).repeat(1, enc_hs.size()[1], 1)), -1)
        ctx_arg2_att = self.w2(ctx_arg2).squeeze()
        ctx_arg2_att.data.masked_fill_(src_mask.data, -float('inf'))
        ctx_arg2_att = F.softmax(ctx_arg2_att, dim=-1)
        ctx2 = torch.bmm(ctx_arg2_att.unsqueeze(1), enc_hs).squeeze()

        return torch.cat((ctx1, ctx2), -1)


def get_vec(arg1s, arg1e, arg2s, arg2e, rel):
    arg1svec = F.softmax(arg1s, dim=-1)
    arg1evec = F.softmax(arg1e, dim=-1)
    arg2svec = F.softmax(arg2s, dim=-1)
    arg2evec = F.softmax(arg2e, dim=-1)
    relvec = F.softmax(rel, dim=-1)
    argvec = arg1svec + arg1evec + arg2svec + arg2evec
    argvec = torch.cat((argvec, relvec), -1)
    return argvec


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, is_bidirectional, drop_out_rate):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.is_bidirectional = is_bidirectional
        self.drop_rate = drop_out_rate
        self.word_embeddings = WordEmbeddings(len(word_vocab), word_embed_dim, drop_rate)
        self.char_embeddings = CharEmbeddings(len(char_vocab), char_embed_dim, drop_rate)
        self.pos_embeddings = POSEmbeddings(len(pos_vocab), pos_tag_dim, drop_rate)
        if enc_type == 'LSTM':
            self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.layers, batch_first=True,
                                bidirectional=self.is_bidirectional)

        self.dropout = nn.Dropout(self.drop_rate)

    def forward(self, words, chars, pos_seq, adv=None, is_training=False):
        src_word_embeds = self.word_embeddings(words)
        if adv is not None:
            batch_len = src_word_embeds.size()[0]
            seq_len = src_word_embeds.size()[1]
            adv = adv.unsqueeze(0).repeat(batch_len, 1, 1)
            adv = torch.index_select(adv.view(-1, word_embed_dim), 0,
                                     words.data.view(-1)).view(batch_len, seq_len, -1)
            src_word_embeds.data = src_word_embeds.data + adv
        char_feature = self.char_embeddings(chars)
        src_pos_embeds = self.pos_embeddings(pos_seq)
        words_input = torch.cat((src_word_embeds, char_feature, src_pos_embeds), -1)
        outputs, hc = self.lstm(words_input)
        outputs = self.dropout(outputs)
        return outputs


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, drop_out_rate, max_length):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.drop_rate = drop_out_rate
        self.max_length = max_length

        if att_type == 0:
            self.attention = Attention(input_dim)
            self.lstm = nn.LSTMCell(10 * self.input_dim, self.hidden_dim)
        elif att_type == 1:
            self.w = nn.Linear(9 * self.input_dim, self.input_dim)
            self.attention = Attention(input_dim)
            self.lstm = nn.LSTMCell(10 * self.input_dim, self.hidden_dim)
        else:
            self.w = nn.Linear(rel_embed_dim + 4 * pointer_net_hidden_size, self.input_dim)
            self.attention1 = Attention(input_dim)
            self.attention2 = Attention(input_dim)
            self.lstm = nn.LSTMCell(rel_embed_dim + 4 * pointer_net_hidden_size + 2 * enc_hidden_size,
                                    self.hidden_dim)

        self.e1_pointer_lstm = nn.LSTM(enc_hidden_size + dec_hidden_size, int(pointer_net_hidden_size/2),
                                       1, batch_first=True, bidirectional=True)
        self.e2_pointer_lstm = nn.LSTM(enc_hidden_size + dec_hidden_size + pointer_net_hidden_size,
                                       int(pointer_net_hidden_size/2), 1, batch_first=True, bidirectional=True)

        self.arg1s_lin = nn.Linear(pointer_net_hidden_size, 1)
        self.arg1e_lin = nn.Linear(pointer_net_hidden_size, 1)
        self.arg2s_lin = nn.Linear(pointer_net_hidden_size, 1)
        self.arg2e_lin = nn.Linear(pointer_net_hidden_size, 1)
        self.sent_att = Sentiment_Attention(enc_hidden_size, 2 * pointer_net_hidden_size)
        self.rel_lin = nn.Linear(dec_hidden_size + 4 * pointer_net_hidden_size + 2 * enc_hidden_size,
                                 len(relnameToIdx))
        self.dropout = nn.Dropout(self.drop_rate)

    def forward(self, prev_tuples, h_prev, enc_hs, src_mask, arg1swts, arg1ewts, arg2swts, arg2ewts,
                is_training=False):
        src_time_steps = enc_hs.size()[1]

        if att_type == 0:
            ctx, attn_weights = self.attention(h_prev[0].squeeze().unsqueeze(1).repeat(1, src_time_steps, 1),
                                                enc_hs, src_mask)
        elif att_type == 1:
            reduce_prev_tuples = self.w(prev_tuples)
            ctx, attn_weights = self.attention(reduce_prev_tuples.unsqueeze(1).repeat(1, src_time_steps, 1),
                                                enc_hs, src_mask)
        else:
            ctx1, attn_weights1 = self.attention1(h_prev[0].squeeze().unsqueeze(1).repeat(1, src_time_steps, 1),
                                               enc_hs, src_mask)
            reduce_prev_tuples = self.w(prev_tuples)
            ctx2, attn_weights2 = self.attention2(reduce_prev_tuples.unsqueeze(1).repeat(1, src_time_steps, 1),
                                               enc_hs, src_mask)
            ctx = torch.cat((ctx1, ctx2), -1)
            attn_weights = (attn_weights1 + attn_weights2) / 2

        s_cur = torch.cat((prev_tuples, ctx), 1)
        hidden, cell_state = self.lstm(s_cur, h_prev)
        hidden = self.dropout(hidden)

        if use_hadamard:
            enc_hs = enc_hs * attn_weights.unsqueeze(2)

        e1_pointer_lstm_input = torch.cat((enc_hs, hidden.unsqueeze(1).repeat(1, src_time_steps, 1)), 2)
        e1_pointer_lstm_out, phc = self.e1_pointer_lstm(e1_pointer_lstm_input)
        e1_pointer_lstm_out = self.dropout(e1_pointer_lstm_out)

        arg1s = self.arg1s_lin(e1_pointer_lstm_out).squeeze()
        arg1s.data.masked_fill_(src_mask.data, -float('inf'))

        arg1e = self.arg1e_lin(e1_pointer_lstm_out).squeeze()
        arg1e.data.masked_fill_(src_mask.data, -float('inf'))

        arg1sweights = F.softmax(arg1s, dim=-1)
        arg1eweights = F.softmax(arg1e, dim=-1)

        arg1sv = torch.bmm(arg1sweights.unsqueeze(1), e1_pointer_lstm_out).squeeze()
        arg1ev = torch.bmm(arg1eweights.unsqueeze(1), e1_pointer_lstm_out).squeeze()
        arg1 = torch.cat((arg1sv, arg1ev), -1)

        e2_pointer_lstm_input = torch.cat((e1_pointer_lstm_input, e1_pointer_lstm_out), 2)
        e2_pointer_lstm_out, phc = self.e2_pointer_lstm(e2_pointer_lstm_input)
        e2_pointer_lstm_out = self.dropout(e2_pointer_lstm_out)

        arg2s = self.arg2s_lin(e2_pointer_lstm_out).squeeze()
        arg2s.data.masked_fill_(src_mask.data, -float('inf'))

        arg2e = self.arg2e_lin(e2_pointer_lstm_out).squeeze()
        arg2e.data.masked_fill_(src_mask.data, -float('inf'))

        arg2sweights = F.softmax(arg2s, dim=-1)
        arg2eweights = F.softmax(arg2e, dim=-1)

        arg2sv = torch.bmm(arg2sweights.unsqueeze(1), e2_pointer_lstm_out).squeeze()
        arg2ev = torch.bmm(arg2eweights.unsqueeze(1), e2_pointer_lstm_out).squeeze()
        arg2 = torch.cat((arg2sv, arg2ev), -1)

        sent_ctx = self.sent_att(arg1, arg2, enc_hs, src_mask)
        rel = self.rel_lin(self.dropout(torch.cat((hidden, arg1, arg2, sent_ctx), -1)))

        if is_training:
            pred_vec = get_vec(arg1s, arg1e, arg2s, arg2e, rel)
            arg1s = F.log_softmax(arg1s, dim=-1)
            arg1e = F.log_softmax(arg1e, dim=-1)
            arg2s = F.log_softmax(arg2s, dim=-1)
            arg2e = F.log_softmax(arg2e, dim=-1)
            rel = F.log_softmax(rel, dim=-1)
            if use_gold_location:
                arg1sv = torch.bmm(arg1swts.unsqueeze(1), e1_pointer_lstm_out).squeeze()
                arg1ev = torch.bmm(arg1ewts.unsqueeze(1), e1_pointer_lstm_out).squeeze()
                arg1 = torch.cat((arg1sv, arg1ev), -1)

                arg2sv = torch.bmm(arg2swts.unsqueeze(1), e2_pointer_lstm_out).squeeze()
                arg2ev = torch.bmm(arg2ewts.unsqueeze(1), e2_pointer_lstm_out).squeeze()
                arg2 = torch.cat((arg2sv, arg2ev), -1)

            return rel.unsqueeze(1), arg1s.unsqueeze(1), arg1e.unsqueeze(1), arg2s.unsqueeze(1), \
                arg2e.unsqueeze(1), (hidden, cell_state), arg1, arg2, pred_vec
        else:
            arg1s = F.softmax(arg1s, dim=-1)
            arg1e = F.softmax(arg1e, dim=-1)
            arg2s = F.softmax(arg2s, dim=-1)
            arg2e = F.softmax(arg2e, dim=-1)
            rel = F.softmax(rel, dim=-1)
            return rel.unsqueeze(1), arg1s.unsqueeze(1), arg1e.unsqueeze(1), arg2s.unsqueeze(1), arg2e.unsqueeze(1), \
                   (hidden, cell_state), arg1, arg2


class Seq2SeqModel(nn.Module):
    def __init__(self):
        super(Seq2SeqModel, self).__init__()
        self.encoder = Encoder(enc_inp_size, int(enc_hidden_size/2), 1, True, drop_rate)
        self.decoder = Decoder(dec_inp_size, dec_hidden_size, 1, drop_rate, max_trg_len)
        self.relation_embeddings = nn.Embedding(len(relnameToIdx), rel_embed_dim)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, src_words_seq, src_mask, src_char_seq, pos_seq, trg_words_seq, trg_seq_len,
                arg1swts, arg1ewts, arg2swts, arg2ewts, adv=None, is_training=False):
        if is_training:
            trg_word_embeds = self.dropout(self.relation_embeddings(trg_words_seq))
        batch_len = src_words_seq.size()[0]
        src_seq_len = src_words_seq.size()[1]
        # trg_seq_len = trg_rel_cnt

        enc_hs = self.encoder(src_words_seq, src_char_seq, pos_seq, adv, is_training)

        h0 = autograd.Variable(torch.FloatTensor(torch.zeros(batch_len, dec_hidden_size))).cuda()
        c0 = autograd.Variable(torch.FloatTensor(torch.zeros(batch_len, dec_hidden_size))).cuda()
        dec_hid = (h0, c0)

        rel_embed = autograd.Variable(torch.FloatTensor(torch.zeros(batch_len, rel_embed_dim))).cuda()
        arg1 = autograd.Variable(torch.FloatTensor(torch.zeros(batch_len, 2 * pointer_net_hidden_size))).cuda()
        arg2 = autograd.Variable(torch.FloatTensor(torch.zeros(batch_len, 2 * pointer_net_hidden_size))).cuda()

        prev_tuples = torch.cat((arg1, arg2, rel_embed), -1)

        if is_training:
            dec_outs = self.decoder(prev_tuples, dec_hid, enc_hs, src_mask,
                                    arg1swts[:, 0, :].squeeze(), arg1ewts[:, 0, :].squeeze(),
                                    arg2swts[:, 0, :].squeeze(), arg2ewts[:, 0, :].squeeze(), is_training)
            pred_vec = dec_outs[8].unsqueeze(1)
        else:
            dec_outs = self.decoder(prev_tuples, dec_hid, enc_hs, src_mask, None, None, None, None, is_training)
        rel = dec_outs[0]
        arg1s = dec_outs[1]
        arg1e = dec_outs[2]
        arg2s = dec_outs[3]
        arg2e = dec_outs[4]
        dec_hid = dec_outs[5]
        arg1 = dec_outs[6]
        arg2 = dec_outs[7]

        topv, topi = rel[:, :, 1:].topk(1)
        topi = torch.add(topi, 1)

        for t in range(1, trg_seq_len):
            if is_training:
                rel_embed = trg_word_embeds[:, t - 1, :].squeeze()
                prev_tuples = torch.cat((arg1, arg2, rel_embed), -1) + prev_tuples
                dec_outs = self.decoder(prev_tuples / (t+1), dec_hid, enc_hs, src_mask,
                                        arg1swts[:, t, :].squeeze(), arg1ewts[:, t, :].squeeze(),
                                        arg2swts[:, t, :].squeeze(), arg2ewts[:, t, :].squeeze(), is_training)
                pred_vec = torch.cat((pred_vec, dec_outs[8].unsqueeze(1)), 1)
            else:
                rel_embed = self.relation_embeddings(topi.squeeze().detach()).squeeze()
                prev_tuples = torch.cat((arg1, arg2, rel_embed), -1) + prev_tuples
                dec_outs = self.decoder(prev_tuples / (t+1), dec_hid, enc_hs, src_mask,
                                        None, None, None, None, is_training)

            cur_rel = dec_outs[0]
            cur_arg1s = dec_outs[1]
            cur_arg1e = dec_outs[2]
            cur_arg2s = dec_outs[3]
            cur_arg2e = dec_outs[4]
            dec_hid = dec_outs[5]
            arg1 = dec_outs[6]
            arg2 = dec_outs[7]

            rel = torch.cat((rel, cur_rel), 1)
            arg1s = torch.cat((arg1s, cur_arg1s), 1)
            arg1e = torch.cat((arg1e, cur_arg1e), 1)
            arg2s = torch.cat((arg2s, cur_arg2s), 1)
            arg2e = torch.cat((arg2e, cur_arg2e), 1)

            topv, topi = cur_rel[:, :, 1:].topk(1)
            topi = torch.add(topi, 1)

        if is_training:
            rel = rel.view(-1, len(relnameToIdx))
            arg1s = arg1s.view(-1, src_seq_len)
            arg1e = arg1e.view(-1, src_seq_len)
            arg2s = arg2s.view(-1, src_seq_len)
            arg2e = arg2e.view(-1, src_seq_len)
            return rel, arg1s, arg1e, arg2s, arg2e, pred_vec
        else:
            return rel, arg1s, arg1e, arg2s, arg2e


def get_model(model_id):
    if model_id == 1:
        return Seq2SeqModel()


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 1:
        torch.cuda.manual_seed_all(seed)


def predict(samples, model, model_id):
    pred_batch_size = batch_size
    batch_count = math.ceil(len(samples) / pred_batch_size)
    move_last_batch = False
    if len(samples) - batch_size * (batch_count - 1) == 1:
        move_last_batch = True
        batch_count -= 1
    rel = list()
    arg1s = list()
    arg1e = list()
    arg2s = list()
    arg2e = list()
    model.eval()
    set_random_seeds(random_seed)
    start_time = datetime.datetime.now()
    for batch_idx in tqdm(range(0, batch_count)):
        batch_start = batch_idx * pred_batch_size
        batch_end = min(len(samples), batch_start + pred_batch_size)
        if batch_idx == batch_count - 1 and move_last_batch:
            batch_end = len(samples)

        cur_batch = samples[batch_start:batch_end]
        cur_samples_input = get_batch_data(cur_batch, False)

        src_words_seq = torch.from_numpy(cur_samples_input['src_words'].astype('long'))
        src_words_mask = torch.from_numpy(cur_samples_input['src_words_mask'].astype('bool'))
        trg_words_seq = torch.from_numpy(cur_samples_input['decoder_input'].astype('long'))
        src_chars_seq = torch.from_numpy(cur_samples_input['src_chars'].astype('long'))
        src_pos_tags = torch.from_numpy(cur_samples_input['src_pos_tags'].astype('long'))

        if torch.cuda.is_available():
            src_words_seq = src_words_seq.cuda()
            src_words_mask = src_words_mask.cuda()
            trg_words_seq = trg_words_seq.cuda()
            src_chars_seq = src_chars_seq.cuda()
            src_pos_tags = src_pos_tags.cuda()

        src_words_seq = autograd.Variable(src_words_seq)
        src_words_mask = autograd.Variable(src_words_mask)
        trg_words_seq = autograd.Variable(trg_words_seq)
        src_chars_seq = autograd.Variable(src_chars_seq)
        src_pos_tags = autograd.Variable(src_pos_tags)

        with torch.no_grad():
            if model_id == 1:
                outputs = model(src_words_seq, src_words_mask, src_chars_seq, src_pos_tags, trg_words_seq,
                                max_trg_len, None, None, None, None, None, False)

        rel += list(outputs[0].data.cpu().numpy())
        arg1s += list(outputs[1].data.cpu().numpy())
        arg1e += list(outputs[2].data.cpu().numpy())
        arg2s += list(outputs[3].data.cpu().numpy())
        arg2e += list(outputs[4].data.cpu().numpy())
        model.zero_grad()

    end_time = datetime.datetime.now()
    custom_print('Prediction time:', end_time - start_time)
    return rel, arg1s, arg1e, arg2s, arg2e


def train_model(model_id, train_samples, dev_samples, test_samples, best_model_file):
    train_size = len(train_samples)
    batch_count = int(math.ceil(train_size/batch_size))
    move_last_batch = False
    if len(train_samples) - batch_size * (batch_count - 1) == 1:
        move_last_batch = True
        batch_count -= 1
    custom_print(batch_count)
    model = get_model(model_id)
    # for name, param in model.named_parameters():
    #     print(name)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    custom_print('Parameters size:', pytorch_total_params)

    custom_print(model)
    if torch.cuda.is_available():
        model.cuda()
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    rel_criterion = nn.NLLLoss(ignore_index=0)
    pointer_criterion = nn.NLLLoss(ignore_index=-1)
    vec_criterion = nn.MSELoss()

    custom_print('weight factor:', wf)
    optimizer = optim.Adam(model.parameters(), weight_decay=0.00001)
    custom_print(optimizer)

    best_dev_acc = -1.0
    best_epoch_idx = -1
    best_epoch_seed = -1
    for epoch_idx in range(0, num_epoch):
        model.train()
        model.zero_grad()
        custom_print('Epoch:', epoch_idx + 1)
        cur_seed = random_seed + epoch_idx + 1

        set_random_seeds(cur_seed)
        cur_shuffled_train_data = shuffle_data(train_samples)
        start_time = datetime.datetime.now()
        train_loss_val = 0.0

        for batch_idx in tqdm(range(0, batch_count)):
            batch_start = batch_idx * batch_size
            batch_end = min(len(cur_shuffled_train_data), batch_start + batch_size)
            if batch_idx == batch_count - 1 and move_last_batch:
                batch_end = len(cur_shuffled_train_data)

            cur_batch = cur_shuffled_train_data[batch_start:batch_end]
            cur_samples_input = get_batch_data(cur_batch, True)

            src_words_seq = torch.from_numpy(cur_samples_input['src_words'].astype('long'))
            src_words_mask = torch.from_numpy(cur_samples_input['src_words_mask'].astype('bool'))
            trg_words_seq = torch.from_numpy(cur_samples_input['decoder_input'].astype('long'))
            src_chars_seq = torch.from_numpy(cur_samples_input['src_chars'].astype('long'))
            src_pos_tags = torch.from_numpy(cur_samples_input['src_pos_tags'].astype('long'))

            arg1sweights = torch.from_numpy(cur_samples_input['arg1sweights'].astype('float32'))
            arg1eweights = torch.from_numpy(cur_samples_input['arg1eweights'].astype('float32'))
            arg2sweights = torch.from_numpy(cur_samples_input['arg2sweights'].astype('float32'))
            arg2eweights = torch.from_numpy(cur_samples_input['arg2eweights'].astype('float32'))

            rel = torch.from_numpy(cur_samples_input['rel'].astype('long'))
            arg1s = torch.from_numpy(cur_samples_input['arg1_start'].astype('long'))
            arg1e = torch.from_numpy(cur_samples_input['arg1_end'].astype('long'))
            arg2s = torch.from_numpy(cur_samples_input['arg2_start'].astype('long'))
            arg2e = torch.from_numpy(cur_samples_input['arg2_end'].astype('long'))
            trg_vec = torch.from_numpy(cur_samples_input['target_vec'].astype('float32'))
            trg_vec_mask = torch.from_numpy(cur_samples_input['target_vec_mask'].astype('bool'))

            src_words_seq = autograd.Variable(src_words_seq.cuda())
            src_words_mask = autograd.Variable(src_words_mask.cuda())
            trg_words_seq = autograd.Variable(trg_words_seq.cuda())
            src_chars_seq = autograd.Variable(src_chars_seq.cuda())
            src_pos_tags = autograd.Variable(src_pos_tags.cuda())

            arg1sweights = autograd.Variable(arg1sweights.cuda())
            arg1eweights = autograd.Variable(arg1eweights.cuda())
            arg2sweights = autograd.Variable(arg2sweights.cuda())
            arg2eweights = autograd.Variable(arg2eweights.cuda())

            rel = autograd.Variable(rel.cuda())
            arg1s = autograd.Variable(arg1s.cuda())
            arg1e = autograd.Variable(arg1e.cuda())
            arg2s = autograd.Variable(arg2s.cuda())
            arg2e = autograd.Variable(arg2e.cuda())
            trg_vec = autograd.Variable(trg_vec.cuda())
            trg_vec_mask = autograd.Variable(trg_vec_mask.cuda())
            trg_seq_len = rel.size()[1]
            if model_id == 1:
                outputs = model(src_words_seq, src_words_mask, src_chars_seq, src_pos_tags, trg_words_seq,
                                trg_seq_len, arg1sweights, arg1eweights, arg2sweights, arg2eweights, None, True)

            rel = rel.view(-1, 1).squeeze()
            arg1s = arg1s.view(-1, 1).squeeze()
            arg1e = arg1e.view(-1, 1).squeeze()
            arg2s = arg2s.view(-1, 1).squeeze()
            arg2e = arg2e.view(-1, 1).squeeze()
            outputs[5].data.masked_fill_(trg_vec_mask.unsqueeze(2).data, 0)
            pred_vec = torch.sum(outputs[5], 1)

            loss = rel_criterion(outputs[0], rel) + \
                   wf * (pointer_criterion(outputs[1], arg1s) + pointer_criterion(outputs[2], arg1e)) + \
                   wf * (pointer_criterion(outputs[3], arg2s) + pointer_criterion(outputs[4], arg2e))

            if use_vec_loss:
                loss = loss + vec_criterion(pred_vec, trg_vec)
            if use_adv:
                model.zero_grad()
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                params = list(model.parameters())
                adv = params[0].grad.data
                adv = adv_eps * math.sqrt(adv.size()[1]) * adv / torch.norm(adv)
                adv_outputs = model(src_words_seq, src_words_mask, src_chars_seq, src_pos_tags, trg_words_seq,
                                trg_seq_len, arg1sweights, arg1eweights, arg2sweights, arg2eweights, adv, True)
                adv_loss = rel_criterion(adv_outputs[0], rel) + \
                   wf * (pointer_criterion(adv_outputs[1], arg1s) + pointer_criterion(adv_outputs[2], arg1e)) + \
                   wf * (pointer_criterion(adv_outputs[3], arg2s) + pointer_criterion(adv_outputs[4], arg2e))
                loss = loss + adv_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            if (batch_idx + 1) % update_freq == 0:
                optimizer.step()
                model.zero_grad()
            train_loss_val += loss.item()

        train_loss_val /= batch_count
        if use_adv:
            train_loss_val /= 2
        end_time = datetime.datetime.now()
        custom_print('Training loss:', train_loss_val)
        custom_print('Training time:', end_time - start_time)

        custom_print('\nDev Results\n')
        set_random_seeds(random_seed)
        dev_preds = predict(dev_samples, model, model_id)

        pred_pos, gt_pos, correct_pos = get_F1(dev_samples, dev_preds)
        custom_print(pred_pos, '\t', gt_pos, '\t', correct_pos)
        p = float(correct_pos) / (pred_pos + 1e-8)
        r = float(correct_pos) / (gt_pos + 1e-8)
        dev_acc = (2 * p * r) / (p + r + 1e-8)
        custom_print('F1:', dev_acc)

        if dev_acc >= best_dev_acc:
            best_epoch_idx = epoch_idx + 1
            best_epoch_seed = cur_seed
            custom_print('model saved......')
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), best_model_file)

        custom_print('\nTest Results\n')
        set_random_seeds(random_seed)
        test_preds = predict(test_samples, model, model_id)

        pred_pos, gt_pos, correct_pos = get_F1(test_samples, test_preds)
        custom_print(pred_pos, '\t', gt_pos, '\t', correct_pos)
        p = float(correct_pos) / (pred_pos + 1e-8)
        r = float(correct_pos) / (gt_pos + 1e-8)
        test_acc = (2 * p * r) / (p + r + 1e-8)
        custom_print('Test F1:', test_acc)

        custom_print('\n\n')
        if epoch_idx + 1 - best_epoch_idx >= early_stop_cnt:
            break

    custom_print('*******')
    custom_print('Best Epoch:', best_epoch_idx)
    custom_print('Best Epoch Seed:', best_epoch_seed)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    random_seed = int(sys.argv[2])
    n_gpu = torch.cuda.device_count()
    set_random_seeds(random_seed)

    src_data_folder = sys.argv[3]
    trg_data_folder = sys.argv[4]
    if not os.path.exists(trg_data_folder):
        os.mkdir(trg_data_folder)
    model_name = 1
    job_mode = sys.argv[5]
    use_data_aug = True  # bool(int(sys.argv[6]))
    use_gold_location = False   # bool(int(sys.argv[7]))
    use_adv = False    # bool(int(sys.argv[8]))
    adv_eps = 0.01   # float(sys.argv[9])
    use_vec_loss = False
    # run = sys.argv[5]
    batch_size = 16
    num_epoch = 150

    max_src_len = 100
    max_trg_len = 10
    embedding_file = 'cased_glove300.txt'
    update_freq = 1
    wf = 1.0
    att_type = 2

    use_hadamard = False  # bool(int(sys.argv[13]))
    enc_type = ['LSTM', 'GCN', 'LSTM-GCN'][0]

    word_embed_dim = 300
    word_min_freq = 2

    char_embed_dim = 25
    pos_tag_dim = 25
    char_feature_size = 25
    conv_filter_size = 3
    max_word_len = 10
    rel_embed_dim = 25

    enc_inp_size = word_embed_dim + char_feature_size + pos_tag_dim
    enc_hidden_size = 300
    dec_inp_size = enc_hidden_size
    dec_hidden_size = dec_inp_size
    pointer_net_hidden_size = 2 * enc_hidden_size

    drop_rate = 0.5
    early_stop_cnt = 5
    Sample = recordclass("Sample", "Id SrcLen SrcWords TrgLen TrgRels TrgPointers")
    rel_file = os.path.join(src_data_folder, 'relations.txt')
    relnameToIdx, relIdxToName = get_relations(rel_file)

    # train a model
    if job_mode == 'train':
        logger = open(os.path.join(trg_data_folder, 'training.log'), 'w')
        custom_print(sys.argv)
        custom_print(max_src_len, max_trg_len, drop_rate)
        custom_print(enc_type)
        custom_print('loading data......')
        model_file_name = os.path.join(trg_data_folder, 'model.h5py')

        src_train_file = os.path.join(src_data_folder, 'train.sent')
        trg_train_file = os.path.join(src_data_folder, 'train.pointer')
        train_data = read_data(src_train_file, trg_train_file, 1)

        src_dev_file = os.path.join(src_data_folder, 'dev.sent')
        trg_dev_file = os.path.join(src_data_folder, 'dev.pointer')
        dev_data = read_data(src_dev_file, trg_dev_file, 2)

        src_test_file = os.path.join(src_data_folder, 'test.sent')
        trg_test_file = os.path.join(src_data_folder, 'test.pointer')
        test_data = read_data(src_test_file, trg_test_file, 3)

        custom_print('Training data size:', len(train_data))
        custom_print('Development data size:', len(dev_data))
        custom_print('Test data size:', len(test_data))

        all_data = train_data + dev_data + test_data

        custom_print("preparing vocabulary......")
        save_vocab = os.path.join(trg_data_folder, 'vocab.pkl')

        custom_print("getting pos tags......")
        pos_vocab = build_tags(src_train_file, src_dev_file, src_test_file)

        word_vocab, char_vocab, word_embed_matrix = build_vocab(all_data, save_vocab, embedding_file)

        custom_print("Training started......")
        train_model(model_name, train_data, dev_data, test_data, model_file_name)
        logger.close()

    if job_mode == 'test':
        logger = open(os.path.join(trg_data_folder, 'test.log'), 'w')
        custom_print(sys.argv)
        custom_print("loading word vectors......")
        vocab_file_name = os.path.join(trg_data_folder, 'vocab.pkl')
        word_vocab, char_vocab, pos_vocab = load_vocab(vocab_file_name)

        # word_embed_matrix = np.zeros((len(word_vocab), word_embed_dim), dtype=np.float32)
        custom_print('vocab size:', len(word_vocab))

        model_file = os.path.join(trg_data_folder, 'model.h5py')

        best_model = get_model(model_name)
        custom_print(best_model)
        if torch.cuda.is_available():
            best_model.cuda()
        if n_gpu > 1:
            best_model = torch.nn.DataParallel(best_model)
        best_model.load_state_dict(torch.load(model_file))

        custom_print('\nTest Results\n')
        src_test_file = os.path.join(src_data_folder, 'test.sent')
        trg_test_file = os.path.join(src_data_folder, 'test.pointer')
        # adj_test_file = os.path.join(src_data_folder, 'test.dep')
        test_data = read_data(src_test_file, trg_test_file, 3)

        reader = open(os.path.join(src_data_folder, 'test.tup'))
        test_gt_lines = reader.readlines()
        reader.close()

        print('Test size:', len(test_data))
        set_random_seeds(random_seed)
        test_preds = predict(test_data, best_model, model_name)
        pred_pos, gt_pos, correct_pos = get_F1(test_data, test_preds)
        custom_print(pred_pos, '\t', gt_pos, '\t', correct_pos)
        p = float(correct_pos) / (pred_pos + 1e-8)
        r = float(correct_pos) / (gt_pos + 1e-8)
        test_acc = (2 * p * r) / (p + r + 1e-8)
        custom_print('P:', round(p, 3))
        custom_print('R:', round(r, 3))
        custom_print('F1:', round(test_acc, 3))
        # write_test_res(test_data, test_preds, os.path.join(trg_data_folder, 'test.out'))

        write_test_res(src_test_file,test_gt_lines,test_data, test_preds, os.path.join(trg_data_folder, 'test.out'))

        logger.close()
