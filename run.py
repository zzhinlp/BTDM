import argparse
import os
from transformers.models.bert.modeling_bert import BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from bert4keras.tokenizers import Tokenizer
from models.model import BTDM
from utils.util import *
from tqdm import tqdm
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--device_id', default="0", type=str, help="GPU index")
parser.add_argument('--dataset', default='WebNLG_star', type=str, help="NYT, WebNLG, NYT*, WebNLG*")
parser.add_argument('--train', default="train", type=str)
parser.add_argument('--num_epochs', default=200, type=int, help="number of epochs")
parser.add_argument('--multi_gpu', action='store_true', help="ensure multi-gpu training")
parser.add_argument('--max_len', default=100, type=int)
parser.add_argument('--warmup', default=0.0, type=float)
parser.add_argument('--bert_vocab_path', default="./pretrained/bert-base-cased/vocab.txt", type=str)
parser.add_argument('--bert_config_path', default="./pretrained/bert-base-cased/config.json", type=str)
parser.add_argument('--bert_model_path', default="./pretrained/bert-base-cased/pytorch_model.bin", type=str)


def train(args):
    root_path = os.path.abspath(".")
    data_path = os.path.join(root_path, "datasets")
    exout_path = os.path.join(root_path, "output", args.dataset, args.file_id)
    train_path = os.path.join(data_path, args.dataset, "train.json")
    dev_path = os.path.join(data_path, args.dataset, "dev.json")
    test_path = os.path.join(data_path, args.dataset, "test.json")
    rel2id_path = os.path.join(data_path, args.dataset, "rel2id.json")
    test_pred_path = os.path.join(exout_path, "test_pred.json")
    dev_pred_path = os.path.join(exout_path, "dev_pred.json")
    if not os.path.exists(exout_path):
        os.makedirs(exout_path)

    train_data = json.load(open(train_path, 'r', encoding='utf-8'))
    valid_data = json.load(open(dev_path, 'r', encoding='utf-8'))
    test_data = json.load(open(test_path, 'r', encoding='utf-8'))
    id2predicate, predicate2id = json.load(open(rel2id_path))
    tokenizer = Tokenizer(args.bert_vocab_path)
    config = BertConfig.from_pretrained(args.bert_config_path)
    config.num_p = len(id2predicate)
    torch.cuda.set_device(int(args.device_id))
    train_model = BTDM.from_pretrained(pretrained_model_name_or_path=args.bert_model_path, config=config)
    train_model.to("cuda")
    batch_size = 18 if args.datasets in ['NYT','NYT_star'] else 6
    dataloader = data_generator(args, train_data, batch_size, tokenizer, predicate2id, id2predicate)

    t_total = len(dataloader) * args.num_epochs

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in train_model.named_parameters() if "bert." in n],
            "weight_decay": 0.0,
            "lr": 3e-5,
        },
        {
            "params": [p for n, p in train_model.named_parameters() if "bert." not in n],
            "weight_decay": 0.0,
            "lr": (3e-5)*3,
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup * t_total, num_training_steps=t_total
    )
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0

    for param in train_model.parameters():
        mulValue = np.prod(param.size())
        Total_params += mulValue
        if param.requires_grad:
            Trainable_params += mulValue
        else:
            NonTrainable_params += mulValue
    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')
    step = 0
    binary_crossentropy = Loss()
    for epoch in range(args.num_epochs):
        train_model.train()
        epoch_loss = 0
        with tqdm(total=dataloader.__len__(), ncols=80) as t:
            t.set_description('epoch:{}/{}'.format(epoch + 1, args.num_epochs))
            for i, batch in enumerate(dataloader):
                batch = [torch.tensor(d).to("cuda") for d in batch]
                batch_token_ids, batch_mask, batch_s1_labels, batch_o1_labels, batch_s2_mask, batch_o2_mask, batch_s2_labels, batch_o2_labels, batch_s3_mask, batch_o3_mask, batch_r = batch
                s1_pred, o1_pred, s2_pred, o2_pred, p_pred = train_model(batch_token_ids, batch_mask, batch_s2_mask,
                                                                         batch_o2_mask, batch_s3_mask, batch_o3_mask)

                def get_loss(target,pred,mask):
                    loss = binary_crossentropy(args, targets=target, pred=pred)
                    loss = torch.mean(loss, dim=2).to("cuda")
                    loss = torch.sum(loss * mask).to("cuda") / torch.sum(mask).to("cuda")
                    return loss
                s1_loss = get_loss(target=batch_s1_labels, pred=s1_pred, mask=batch_mask)
                o1_loss = get_loss(target=batch_o1_labels, pred=o1_pred, mask=batch_mask)
                s2_loss = get_loss(target=batch_s2_labels, pred=s2_pred, mask=batch_mask)
                o2_loss = get_loss(target=batch_o2_labels, pred=o2_pred, mask=batch_mask)
                r_loss = binary_crossentropy(args, targets=batch_r, pred=p_pred)
                r_loss = r_loss.mean()
                loss = s1_loss + o1_loss + s2_loss + o2_loss + r_loss
                loss.backward()
                step += 1
                epoch_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(train_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_model.zero_grad()
                t.set_postfix(loss="%.4lf" % (loss.cpu().item()))
                t.update(1)
        f1, precision, recall = evaluate(args, tokenizer, id2predicate, train_model, valid_data, dev_pred_path)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(train_model.state_dict(), os.path.join(exout_path, "pytorch_model.bin"))

    train_model.load_state_dict(torch.load(os.path.join(exout_path, "pytorch_model.bin"), map_location="cuda"))
    evaluate(args, tokenizer, id2predicate, train_model, test_data, test_pred_path)


def test(args):
    root_path = os.path.abspath(".")
    torch.cuda.set_device(int(args.device_id))
    data_path = os.path.join(root_path, "datasets")
    test_path = os.path.join(data_path, args.dataset, "test.json")
    exout_path = os.path.join(root_path, "exout", args.dataset)
    test_pred_path = os.path.join(exout_path, "test_pred.json")
    rel2id_path = os.path.join(data_path, args.dataset, "rel2id.json")
    test_data = json.load(open(test_path, encoding="utf-8"))
    id2predicate, predicate2id = json.load(open(rel2id_path))
    config = BertConfig.from_pretrained(args.bert_config_path)
    tokenizer = Tokenizer(args.bert_vocab_path)
    config.num_p = len(id2predicate)
    model = BTDM.from_pretrained(pretrained_model_name_or_path=args.bert_model_path, config=config)
    model.to("cuda")

    model.load_state_dict(torch.load(r"", map_location="cuda"))
    evaluate(args, tokenizer, id2predicate, model, test_data, test_pred_path)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.train == "train":
        train(args)
    else:
        test(args)
