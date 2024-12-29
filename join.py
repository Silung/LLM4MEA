import os
import pickle
import random

root = '/data/zhaoshilong/REA_with_llm/gen_data/steam/bert_5001_100'
# paths = [f'llm_seq{i}_dataset.pkl' for i in range(15,20)]
paths = ['llm_seq10_dataset.pkl', 'random0_dataset.pkl']
print(root)
print(paths)
datasets = []
for path in paths:
    with open(os.path.join(root, path), 'rb') as f:
         datasets.append(pickle.load(f))
    
k=5001
seqs, logits, candidates = [], [], []
for dataset in datasets:
    seqs += dataset['seqs']
    logits += dataset['logits']
    candidates += dataset['candidates']
    
combined = list(zip(seqs, logits, candidates))
random.shuffle(combined)
seqs_shuffled, logits_shuffled, candidates_shuffled = zip(*combined)
seqs = list(seqs_shuffled)
logits = list(logits_shuffled)
candidates = list(candidates_shuffled)

dataset3 = {'seqs':seqs[:k], 'logits':logits[:k], 'candidates':candidates[:k]}
print(f'len(dataset3):{len(dataset3)}')
print(f'len(dataset3["seqs"]):{len(dataset3["seqs"])}')

with open(os.path.join(root, 'llm_seq61_dataset.pkl'), 'wb') as f:
    pickle.dump(dataset3, f)
