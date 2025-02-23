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

    all_overlap_ratios = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            if model_code == 'bert':
                seqs, seq_dis = batch
                seqs = seqs.to(args.device)
                seq_dis = seq_dis.to(args.device)
                scores1 = model(seqs)[:, -1, :]  # 取最后一个时间步的输出
                scores2 = model(seq_dis)[:, -1, :]
            elif model_code == 'sas':
                seqs, seq_dis = batch
                seqs = seqs.to(args.device)
                seq_dis = seq_dis.to(args.device)
                scores1 = model(seqs)[:, -1, :]
                scores2 = model(seq_dis)[:, -1, :]
            elif model_code == 'narm':
                seqs, seq_dis, lengths = batch
                seqs = seqs.to(args.device)
                seq_dis = seq_dis.to(args.device)
                lengths = lengths.flatten()
                scores1 = model(seqs, lengths)
                scores2 = model(seq_dis, lengths)

            _, top100_1 = torch.topk(scores1, 100, dim=1)
            _, top100_2 = torch.topk(scores2, 100, dim=1)
            
            # 计算重叠率
            for i in range(len(top100_1)):
                set1 = set(top100_1[i].cpu().numpy())
                set2 = set(top100_2[i].cpu().numpy())
                overlap_ratio = len(set1.intersection(set2)) / 100
                all_overlap_ratios.append(overlap_ratio)
            
    return sum(all_overlap_ratios) / len(all_overlap_ratios)


if __name__ == "__main__":
    set_template(args, 'test')
    
    model_codes = ['narm','sas', 'bert']
    
    similarities = {}
    for model_code in model_codes:
        args.model_code = model_code
        model_root = f'experiments/{model_code}/steam'
        for i in range(-1, 0):
            args.dis_loc = i
            similarity = test(args, model_root, model_code)
            similarities[model_code, i] = similarity
        
    print(similarities)

# beauty
# {('sas', -4): 0.8969718589966289, ('sas', -3): 0.9011422960274107, ('sas', -2): 0.8965591905732346, ('sas', -1): 0.722083229751909, ('bert', -4): 0.8749689255705165, ('bert', -3): 0.8758598916123869, ('bert', -2): 0.8736158206135272, ('bert', -1): 0.8688614329040882, ('narm', -4): 0.7602853875602779, ('narm', -3): 0.748054492119529, ('narm', -2): 0.7126990006463517, ('narm', -1): 0.610931984288779}
# games
# {('narm', -4): 0.6692945025731962, ('narm', -3): 0.6306267680038229, ('narm', -2): 0.5670611090283225, ('narm', -1): 0.4282556831737135, ('sas', -4): 0.8494185610578825, ('sas', -3): 0.8404403394567195, ('sas', -2): 0.8416383218022313, ('sas', -1): 0.422428001772272, ('bert', -4): 0.8448396441838926, ('bert', -3): 0.8444337275484652, ('bert', -2): 0.841368733171991, ('bert', -1): 0.8344787157901756}
# steam
# {('narm', -4): 0.7941801029466076, ('narm', -3): 0.7756455093828144, ('narm', -2): 0.7311854415886253, ('narm', -1): 0.5583914725206608, ('sas', -4): 0.6768304727059317, ('sas', -3): 0.6681466303184157, ('sas', -2): 0.658995851044066, ('sas', -1): 0.45234616281362455, ('bert', -4): 0.8847090350392575, ('bert', -3): 0.8609422434255223, ('bert', -2): 0.8567687465252741, ('bert', -1): 0.5595729086333117}




{('narm', -1): 0.7401474170934267, ('sas', -1): 0.8006451051558807, ('bert', -1): 0.985658529309536}
{('narm', -1): 0.6830544289560684, ('sas', -1): 0.6651235472546791, ('bert', -1): 0.9824215261921532}
{('narm', -1): 0.703942554298199, ('sas', -1): 0.7398685665776363, ('bert', -1): 0.6570700240927145}