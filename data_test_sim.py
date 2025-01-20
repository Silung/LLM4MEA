from datasets import DATASETS
from config import STATE_DICT_KEY
import torch
import torch.nn.functional as F
from model import *
from dataloader import *
from utils import *
import os
import numpy as np
from tqdm import tqdm

def test(args, model_root, model_code):
    fix_random_seed_as(args.model_init_seed)
    _, _, test_loader = dataloader_factory(args)
    
    # 初始化模型
    if model_code == 'bert':
        model = BERT(args)
    elif model_code == 'sas':
        model = SASRec(args)
    elif model_code == 'narm':
        model = NARM(args)
    
    # 加载模型参数
    state_dict = torch.load(os.path.join(model_root, 'models', 'best_acc_model.pth'), map_location=args.device)
    model.load_state_dict(state_dict.get(STATE_DICT_KEY))
    model = model.to(args.device)
    model.eval()

    all_scores = []
    all_similarity1 = []
    all_similarity2 = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            if model_code == 'bert':
                seqs, candidates, labels = batch
                seqs = seqs.to(args.device)
                scores1 = model(seqs)[:, -1, :]  # 取最后一个时间步的输出
            elif model_code == 'sas':
                seqs, candidates, labels = batch
                seqs = seqs.to(args.device)
                scores1 = model(seqs)[:, -1, :]
            elif model_code == 'narm':
                seqs, lengths, candidates, labels = batch
                seqs = seqs.to(args.device)
                lengths = lengths.flatten()
                scores1 = model(seqs, lengths)
            
            # 生成随机item id (范围1-54542)
            random_items = torch.randint(1, 54543, (scores1.shape[0],)).to(args.device)
            random_items = torch.randint(1, 23465, (scores1.shape[0],)).to(args.device)
            seqs_short = seqs.clone()
            seqs_short[:, -1] = random_items

            if model_code == 'bert':
                scores2 = model(seqs_short)[:, -1, :]  # 取最后一个时间步的输出
            elif model_code == 'sas':
                scores2 = model(seqs_short)[:, -1, :]
            elif model_code == 'narm':
                scores2 = model(seqs_short, lengths)

            seqs_long = seqs.clone()
            seqs_long[:, -3] = random_items
            if model_code == 'bert':
                scores3 = model(seqs_long)[:, -1, :]  # 取最后一个时间步的输出
            elif model_code == 'sas':
                scores3 = model(seqs_long)[:, -1, :]
            elif model_code == 'narm':
                scores3 = model(seqs_long, lengths)
            
            # 计算余弦相似度
            scores1_norm = F.normalize(scores1, p=2, dim=1)
            scores2_norm = F.normalize(scores2, p=2, dim=1)
            scores3_norm = F.normalize(scores3, p=2, dim=1)
            # 对每个batch中的样本计算其对应输出向量的余弦相似度
            similarity1 = torch.sum(scores1_norm * scores2_norm, dim=1)
            similarity2 = torch.sum(scores1_norm * scores3_norm, dim=1)
            
            all_similarity1 += similarity1.detach().cpu().numpy().tolist()
            all_similarity2 += similarity2.detach().cpu().numpy().tolist()
            
    return sum(all_similarity1) / len(all_similarity1), sum(all_similarity2) / len(all_similarity2)


if __name__ == "__main__":
    set_template(args, 'test')
    
    # 加载第一个模型(NARM)
    args.model_code = 'narm'
    model_root = input("输入NARM模型路径:\n")
    similarity1, similarity2 = test(args, model_root, 'narm')
    print(f'short: {similarity1}, long: {similarity2} ')
    
    args.model_code = 'sas'
    model_root = input("输入SASRec模型路径:\n")
    similarity1, similarity2 = test(args, model_root, 'sas')
    print(f'short: {similarity1}, long: {similarity2} ')

    args.model_code = 'bert'
    model_root = input("输入BERT模型路径:\n")
    similarity1, similarity2 = test(args, model_root, 'bert')
    print(f'short: {similarity1}, long: {similarity2} ')
