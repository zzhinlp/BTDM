#! -*- coding:utf-8 -*-
import numpy as np
import random
import torch
import json
from tqdm import tqdm
import torch.nn as nn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def mat_padding(inputs, dim=0, length=None, padding=0):
    if not type(inputs[0]) is np.ndarray:
        inputs = [np.array(i) for i in inputs]

    if length is None:
        length = max([x.shape[dim] for x in inputs])
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        pad_width[0] = (0, length - x.shape[dim])
        pad_width[1] = (0, length - x.shape[dim])
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)
    return np.array(outputs)


def tuple_mat_padding(inputs, dim=1, length=None, padding=0):
    if not type(inputs[0]) is np.ndarray:
        inputs = [np.array(i) for i in inputs]

    if length is None:
        length = max([x.shape[dim] for x in inputs])
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        pad_width[1] = (0, length - x.shape[dim])
        pad_width[2] = (0, length - x.shape[dim])
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)
    return np.array(outputs)


def sequence_padding(inputs, dim=0, length=None, padding=0):
    if not type(inputs[0]) is np.ndarray:
        inputs = [np.array(i) for i in inputs]

    if length is None:
        length = max([x.shape[dim] for x in inputs])
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        pad_width[dim] = (0, length - x.shape[dim])
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)
    return np.array(outputs)


def judge(ex):
    for s, p, o in ex["triple_list"]:
        if s == '' or o == '' or s not in ex["text"] or o not in ex["text"]:
            return False
    return True


class DataGenerator(object):
    """数据生成器模版
    """

    def __init__(self, data, batch_size=32, buffer_size=None):
        self.data = data
        self.batch_size = batch_size
        if hasattr(self.data, '__len__'):
            self.steps = len(self.data) // self.batch_size
            if len(self.data) % self.batch_size != 0:
                self.steps += 1
        else:
            self.steps = None
        self.buffer_size = buffer_size or batch_size * 1000

    def __len__(self):
        return self.steps

    def sample(self, random=True):

        if random:
            if self.steps is None:
                def generator():
                    caches, isfull = [], False
                    for d in self.data:
                        caches.append(d)
                        if isfull:
                            i = np.random.randint(len(caches))
                            yield caches.pop(i)
                        elif len(caches) == self.buffer_size:
                            isfull = True
                    while caches:
                        i = np.random.randint(len(caches))
                        yield caches.pop(i)

            else:
                def generator():
                    indices = list(range(len(self.data)))
                    np.random.shuffle(indices)
                    for i in indices:
                        yield self.data[i]

            data = generator()
        else:
            data = iter(self.data)

        d_current = next(data)
        for d_next in data:
            yield False, d_current
            d_current = d_next

        yield True, d_current

    def __iter__(self, random=False):
        raise NotImplementedError

    def forfit(self):
        for d in self.__iter__(True):
            yield d


def search(pattern, sequence):
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


def judge(ex):
    for s, p, o in ex["triple_list"]:
        if s == '' or o == '' or s not in ex["text"] or o not in ex["text"]:
            return False
    return True


def extract_spoes(args, tokenizer, id2predicate, model, text, entity_start=0.5, entity_end=0.5, p_num=0.5):
    # sigmoid=nn.Sigmoid()
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.to("cuda")
    tokens = tokenizer.tokenize(text, max_length=args.max_len)  # tokenizer 是分词 tokens
    mapping = tokenizer.rematch(text, tokens)
    token_ids, _, mask = tokenizer.encode(text, max_length=args.max_len)
    # 获取BERT表示
    model.eval()
    with torch.no_grad():
        head, tail, rel, cls = model.get_embed(torch.tensor([token_ids]).to("cuda"), torch.tensor([mask]).to("cuda"))
        head = head.cpu().detach().numpy()
        tail = tail.cpu().detach().numpy()
        rel = rel.cpu().detach().numpy()
        cls = cls.cpu().detach().numpy()

    def get_entity(entity_pred):
        start = np.where(entity_pred[0, :, 0] > entity_start)[0]
        end = np.where(entity_pred[0, :, 1] > entity_end)[0]
        entity = []
        for i in start:
            j = end[end >= i]
            if len(j) > 0:
                j = j[0]
                entity.append((i, j))
        return entity

    model.eval()
    with torch.no_grad():
        s1_preds = model.s_pred(torch.tensor(head).to("cuda"), torch.tensor(cls).to("cuda"))
        o1_preds = model.o_pred(torch.tensor(tail).to("cuda"), torch.tensor(cls).to("cuda"))

        s1_preds = s1_preds.cpu().detach().numpy()
        o1_preds = o1_preds.cpu().detach().numpy()

        s1_preds[:, 0, :], s1_preds[:, -1, :] = 0.0, 0.0
        o1_preds[:, 0, :], o1_preds[:, -1, :] = 0.0, 0.0

    s1 = get_entity(s1_preds)
    o1 = get_entity(o1_preds)

    pairs_0 = []
    for s in s1:
        for o in o1:
            pairs_0.append((s[0], s[1], o[0], o[1]))

    pairs_1 = []
    for s in s1:

        s2_mask = np.zeros(len(token_ids)).astype(np.int)
        s2_mask[s[0]] = 1
        s2_mask[s[1]] = 1

        model.eval()
        with torch.no_grad():
            o2_pred = model.o_pred_from_s(torch.tensor(head).to("cuda"), torch.tensor(tail).to("cuda"),
                                          torch.tensor([s2_mask]).to("cuda"), torch.tensor(rel).to("cuda"),
                                          cls=torch.tensor(cls).to("cuda"))
            o2_pred = o2_pred.cpu().detach().numpy()  # [1,L,2]
            o2_pred[:, 0, :], o2_pred[:, -1, :] = 0.0, 0.0
        objects2 = get_entity(o2_pred)
        if objects2:
            for o in objects2:
                pairs_1.append((s[0], s[1], o[0], o[1]))

    pairs_2 = []
    for o in o1:
        o2_mask = np.zeros(len(token_ids)).astype(np.int)
        o2_mask[o[0]] = 1
        o2_mask[o[1]] = 1

        model.eval()
        with torch.no_grad():
            s2_pred = model.s_pred_from_o(torch.tensor(head).to("cuda"), torch.tensor(tail).to("cuda"),
                                          torch.tensor([o2_mask]).to("cuda"), torch.tensor(rel).to("cuda"),
                                          cls=torch.tensor(cls).to("cuda"))
            s2_pred = s2_pred.cpu().detach().numpy()  # [1,L,2]
            s2_pred[:, 0, :], s2_pred[:, -1, :] = 0.0, 0.0
        subjects2 = get_entity(s2_pred)
        if subjects2:
            for s in subjects2:
                pairs_2.append((s[0], s[1], o[0], o[1]))

    pairs_1 = set(pairs_1)
    pairs_2 = set(pairs_2)

    pairs = list(pairs_1 | pairs_2)

    if pairs:
        s_mask = np.zeros([len(pairs), len(token_ids)]).astype(np.int)
        o_mask = np.zeros([len(pairs), len(token_ids)]).astype(np.int)

        for i, pair in enumerate(pairs):
            s1, s2, o1, o2 = pair
            s_mask[i, s1] = 1
            s_mask[i, s2] = 1
            o_mask[i, o1] = 1
            o_mask[i, o2] = 1

        spoes = []
        rel = np.repeat(rel, len(pairs), 0)

        model.eval()
        with torch.no_grad():
            p_pred = model.p_pred(
                rel=torch.tensor(rel).to("cuda"),
                s_mask=torch.tensor(s_mask).to("cuda"),
                o_mask=torch.tensor(o_mask).to("cuda"),
            )
            p_pred = p_pred.cpu().detach().numpy()

        index, p_index = np.where(p_pred > p_num)
        for i, p in zip(index, p_index):
            s1, s2, o1, o2 = pairs[i]
            spoes.append(
                (
                    (mapping[s1][0], mapping[s2][-1]),
                    p,
                    (mapping[o1][0], mapping[o2][-1])
                )
            )

        return [(text[s[0]:s[1] + 1], id2predicate[str(p)], text[o[0]:o[1] + 1])
                for s, p, o, in spoes]
    else:
        return []

    return loss


class Loss():
    def __call__(self, args, targets, pred, from_logist=False):
        if not from_logist:
            pred = torch.where(pred < 1 - args.min_num, pred, torch.ones(pred.shape).to("cuda") * 1 - args.min_num).to(
                "cuda")
            pred = torch.where(pred > args.min_num, pred, torch.ones(pred.shape).to("cuda") * args.min_num).to("cuda")
            pred = torch.log(pred / (1 - pred))
        relu = nn.ReLU()
        loss = relu(pred) - pred * targets + torch.log(1 + torch.exp(-1 * torch.abs(pred).to("cuda"))).to("cuda")
        return loss


class data_generator(DataGenerator):

    def __init__(self, args, train_data,batch_size,tokenizer, predicate2id, id2predicate):
        super(data_generator, self).__init__(train_data, batch_size)
        self.max_len = args.max_len
        self.tokenizer = tokenizer
        self.predicate2id = predicate2id
        self.id2predicate = id2predicate

    def __iter__(self, is_random=False):
        batch_token_ids, batch_mask = [], []
        batch_s1_labels, batch_o1_labels, batch_s2_mask, batch_o2_mask, = [], [], [], []
        batch_s2_labels, batch_o2_labels, batch_s3_mask, batch_o3_mask, batch_r = [], [], [], [], []
        for is_end, d in self.sample(is_random):
            if judge(d) == False:
                continue
            token_ids, _, mask = self.tokenizer.encode(
                d['text'], max_length=self.max_len
            )
            spoes_s = {}
            spoes_o = {}
            for s, p, o in d['triple_list']:
                s = self.tokenizer.encode(s)[0][1:-1]
                p = self.predicate2id[p]
                o = self.tokenizer.encode(o)[0][1:-1]
                s_idx = search(s, token_ids)
                o_idx = search(o, token_ids)
                if s_idx != -1 and o_idx != -1:
                    s_loc = (s_idx, s_idx + len(s) - 1)
                    o_loc = (o_idx, o_idx + len(o) - 1)
                    if s_loc not in spoes_s:
                        spoes_s[s_loc] = []
                    spoes_s[s_loc].append((o_loc, p))
                    if o_loc not in spoes_o:
                        spoes_o[o_loc] = []
                    spoes_o[o_loc].append((s_loc, p))
            if spoes_s and spoes_o:
                def get_entity1_labels(item, l):
                    res = np.zeros([l, 2])
                    for start, end in item:
                        res[start][0] = 1
                        res[end][1] = 1
                    return res

                s1_labels = get_entity1_labels(spoes_s, len(token_ids))
                o1_labels = get_entity1_labels(spoes_o, len(token_ids))

                def get_entity2_labels_mask(item, l):
                    start, end = random.choice(list(item.keys()))
                    # 构造labels
                    labels = np.zeros((l, 2))
                    if (start, end) in item:
                        for loc, _ in item[
                            (start, end)]:  # loc是指定sub(obj)情况下，obj的开始结束位置，loc[0], 0：表示开始序列位置标1，loc[1], 0：结束位置标1
                            labels[loc[0], 0] = 1
                            labels[loc[1], 1] = 1
                    # 构造mask
                    mask = np.zeros(l)  # sub/obj的掩码，是开始或结尾的位置标1
                    mask[start] = 1
                    mask[end] = 1
                    return labels, mask

                o2_labels, s2_mask = get_entity2_labels_mask(spoes_s, len(token_ids))
                s2_labels, o2_mask = get_entity2_labels_mask(spoes_o, len(token_ids))

                s_loc = random.choice(list(spoes_s.keys()))
                o_loc, _ = random.choice(spoes_s[s_loc])
                r = np.zeros(len(self.id2predicate))
                if s_loc in spoes_s:
                    for loc, the_r in spoes_s[s_loc]:
                        if loc == o_loc:
                            r[the_r] = 1
                s3_mask = np.zeros(len(token_ids))
                o3_mask = np.zeros(len(token_ids))
                s3_mask[s_loc[0]] = 1
                s3_mask[s_loc[1]] = 1
                o3_mask[o_loc[0]] = 1
                o3_mask[o_loc[1]] = 1

                # 构建batch
                batch_token_ids.append(token_ids)
                batch_mask.append(mask)

                batch_s1_labels.append(s1_labels)
                batch_o1_labels.append(o1_labels)

                batch_s2_mask.append(s2_mask)
                batch_o2_mask.append(o2_mask)
                batch_s2_labels.append(s2_labels)
                batch_o2_labels.append(o2_labels)

                batch_s3_mask.append(s3_mask)
                batch_o3_mask.append(o3_mask)
                batch_r.append(r)

                if len(batch_token_ids) == self.batch_size or is_end:  # 输出batch
                    batch_token_ids, batch_mask, \
                    batch_s1_labels, batch_o1_labels, \
                    batch_s2_mask, batch_o2_mask, batch_s2_labels, batch_o2_labels, \
                    batch_s3_mask, batch_o3_mask = \
                        [sequence_padding(i).astype(np.int)
                         for i in [batch_token_ids, batch_mask,
                                   batch_s1_labels, batch_o1_labels,
                                   batch_s2_mask, batch_o2_mask, batch_s2_labels, batch_o2_labels,
                                   batch_s3_mask, batch_o3_mask]]

                    batch_r = np.array(batch_r).astype(np.int)

                    yield [
                        batch_token_ids, batch_mask,
                        batch_s1_labels, batch_o1_labels,
                        batch_s2_mask, batch_o2_mask, batch_s2_labels, batch_o2_labels,
                        batch_s3_mask, batch_o3_mask, batch_r
                    ]
                    batch_token_ids, batch_mask = [], []
                    batch_s1_labels, batch_o1_labels, \
                    batch_s2_mask, batch_o2_mask, batch_s2_labels, batch_o2_labels, \
                    batch_s3_mask, batch_o3_mask, batch_r = [], [], [], [], [], [], [], [], []


def evaluate(args, tokenizer, id2predicate, model, evl_data, evl_path):
    """评估函数，计算f1、precision、recall
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open(evl_path, 'w', encoding='utf-8')
    pbar = tqdm()
    for d in evl_data:
        R = set(extract_spoes(args, tokenizer, id2predicate, model, d['text']))
        T = set([(i[0], i[1], i[2]) for i in d['triple_list']])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        pbar.update()
        pbar.set_description(
            'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
        )
        s = json.dumps({
            'text': d['text'],
            'triple_list': list(T),
            'triple_list_pred': list(R),
            'new': list(R - T),
            'lack': list(T - R),
        }, ensure_ascii=False, indent=4)
        f.write(s + '\n')
    pbar.close()
    f.close()
    return f1, precision, recall
