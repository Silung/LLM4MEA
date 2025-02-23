import os
import pickle
import random

root = '/data/zhaoshilong/REA_with_llm/gen_data/beauty/narm_5001_100'
# paths = [f'llm_seq{i}_dataset.pkl' for i in range(15,20)]
paths = ['llm_seq110_dataset.pkl', 'random0_dataset.pkl']
print(root)
print(paths)
datasets = []
for path in paths:
    with open(os.path.join(root, path), 'rb') as f:
         datasets.append(pickle.load(f))
    
k = 5001  # 总样本数
n = 2512

# 从每个数据集中随机抽取样本
seqs0, logits0, candidates0 = [], [], []
seqs1, logits1, candidates1 = [], [], []

# 从第一个数据集(random0)中抽取n个样本
combined0 = list(zip(datasets[1]['seqs'], datasets[1]['logits'], datasets[1]['candidates']))
random.shuffle(combined0)
seqs0, logits0, candidates0 = zip(*combined0[:n])

# 从第二个数据集(llm_seq10)中抽取剩余样本
combined1 = list(zip(datasets[0]['seqs'], datasets[0]['logits'], datasets[0]['candidates']))
random.shuffle(combined1)
seqs1, logits1, candidates1 = zip(*combined1[:k-n])

# 合并两个数据集的样本
seqs = list(seqs0) + list(seqs1)
logits = list(logits0) + list(logits1)
candidates = list(candidates0) + list(candidates1)

# 打乱合并后的数据
combined = list(zip(seqs, logits, candidates))
random.shuffle(combined)
seqs, logits, candidates = zip(*combined)

dataset3 = {'seqs':seqs[:k], 'logits':logits[:k], 'candidates':candidates[:k]}
print(f'len(dataset3):{len(dataset3)}')
print(f'len(dataset3["seqs"]):{len(dataset3["seqs"])}')

with open(os.path.join(root, 'llm_seq111_dataset.pkl'), 'wb') as f:
    pickle.dump(dataset3, f)
