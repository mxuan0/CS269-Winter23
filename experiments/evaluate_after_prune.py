import torch
import argparse
from Models import BERT
import numpy as np
from PackDataset import packDataset_util_bert

def read_data(file_path):
    import pandas as pd
    data = pd.read_csv(file_path, sep='\t', error_bad_lines=False, engine='python').values.tolist()
    processed_data = []
    for item in data:
        if not np.isnan(item[1]):
            processed_data.append((item[0].lower().strip(), item[1]))
    return processed_data


def get_all_data(base_path):
    import os
    train_path = os.path.join(base_path, 'train1000.tsv')
    dev_path = os.path.join(base_path, 'dev.tsv')
    test_path = os.path.join(base_path, 'test.tsv')
    return read_data(train_path), read_data(dev_path), read_data(test_path)

def get_poison_data(poison_data):
    process_data = []
    for item in poison_data:
        if item[1] != target_label:
            process_data.append((item[0], target_label))
    return process_data

def evaluaion(model, loader):
    model.eval()
    total_number = 0
    total_correct = 0
    with torch.no_grad():
        for padded_text, attention_masks, labels in loader:
            padded_text = padded_text.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)
            output = model(padded_text, attention_masks)  # batch_size, 2
            _, flag = torch.max(output, dim=1)
            total_number += labels.size(0)
            correct = (flag == labels).sum().item()
            total_correct += correct
        acc = total_correct / total_number
        return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--transferdata_path', type=str, default='')
    parser.add_argument('--origdata_path', type=str, default='')
    parser.add_argument('--bert_type', type=str, default='bert-base-uncased')
    parser.add_argument('--output_num', default=2, type=int)
    parser.add_argument('--target_label', default=1, type=int)
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()
    BATCH_SIZE = args.batch_size
    packDataset_util = packDataset_util_bert(args.bert_type)
    target_label = args.target_label

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BERT(output_nums=args.output_num, bert_type=args.bert_type).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.eval()
    # model = torch.load(args.checkpoint_path) #BERT(output_nums=args.output_num, bert_type=args.bert_type).to(device)
    # checkpoint = torch.load(args.checkpoint_path)
    # model.load_state_dict(checkpoint['model_state_dict'])


    clean_train_data, clean_dev_data, clean_test_data = get_all_data(args.origdata_path)
    train_data_poison, dev_data_poison, test_data_poison = get_all_data(args.transferdata_path)

    poison_test_data = get_poison_data(test_data_poison)
    test_loader_poison = packDataset_util.get_loader(poison_test_data, shuffle=False, batch_size=BATCH_SIZE)
    test_loader_clean = packDataset_util.get_loader(clean_test_data, shuffle=False, batch_size=BATCH_SIZE)

    poison_success_rate_test = evaluaion(model, test_loader_poison)
    clean_acc = evaluaion(model, test_loader_clean)
    print('test: {}. clean acc: {}'.format( poison_success_rate_test, clean_acc))