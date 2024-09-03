import os
import pickle

root = 'gen_data/steam/narm_1000_100'
paths = [f'llm_seq{i}_dataset.pkl' for i in range(10,15)]
print(paths)
datasets = []
for path in paths:
    with open(os.path.join(root, path), 'rb') as f:
         datasets.append(pickle.load(f))
    
k=5000
seqs, logits, candidates = [], [], []
for dataset in datasets:
    seqs += dataset['seqs']
    logits += dataset['logits']
    candidates += dataset['candidates']
dataset3 = {'seqs':seqs[:k], 'logits':logits[:k], 'candidates':candidates[:k]}
print(f'len(dataset3):{len(dataset3)}')
print(f'len(dataset3["seqs"]):{len(dataset3["seqs"])}')

with open(os.path.join(root, 'temp.pkl'), 'wb') as f:
    pickle.dump(dataset3, f)
