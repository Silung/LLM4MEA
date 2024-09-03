import os
import json

a1 = ['gru', 'sas', 'bert']
a2 = ['ml-1m', 'steam', 'beauty']
a3 = ['NDCG@10', 'Recall@10', 'Agr@1', 'Agr@10']
a4 = ['target', 'random', 'autoregressive', 'llm_seq']
a1map = {'gru':'GRU4Rec', 'sas':'SASRec', 'bert':'BERT'}
a4map = {'target':'Target', 'random':'Random', 'autoregressive':'DFME', 'llm_seq':'Ours'}

# for i in a3[:2]:
#     for j in a2:
#         for k in a1:
#             f = open(os.path.join('experiments', k, j, 'logs', 'test_metrics.json'), 'r')
#             json_str = "".join(f.readlines())
#             data = json.loads(json_str)
#             # print(f"{i}: \t{data[i]}\t", end='')
#             print(f" {data[i]:.4f} &", end='')
#     print()
    
# print()
r = r"""\begin{tabular}{lccccccccccccc}
\toprule
 & & \multicolumn{4}{c}{ML-1M}   & \multicolumn{4}{c}{Steam} & \multicolumn{4}{c}{Beauty}  \\ 
 \cmidrule(l){3-6} \cmidrule(l){7-10} \cmidrule(l){11-14}
                &&N@10 & R@10 & Agr@1 & Agr@10 & N@10 & R@10 & Agr@1 & Agr@10 & N@10 & R@10 & Agr@1 & Agr@10\\ \midrule
"""

cc = 0
for i in a1:
    r += f'\\multirow{len(a4)}{{*}}{{{a1map[i]}}} '
    for w in a4:
        cc += 1
        r += f'& {a4map[w]} '
        for j in a2:
            for k in a3:
                try:
                    if w == 'target':
                        f = open(os.path.join('experiments', i, j, 'logs', 'test_metrics.json'), 'r')
                        json_str = "".join(f.readlines())
                        data = json.loads(json_str)
                        if k in data:
                            r += f" & {data[k]:.4f}"
                        else:
                            r += f" & - "
                    else:
                        f = open(os.path.join('experiments', 'distillation_rank', f'{i}2{i}_{w}100ranking5000', j, 'logs', 'test_metrics.json'), 'r')    
                        json_str = "".join(f.readlines())
                        data = json.loads(json_str)
                        # print(f"{i}: \t{data[i]}\t", end='')
                        r += f" & {data[k]:.4f}"
                except:
                    # print(os.path.join('experiments', 'distillation_rank', f'{i}2{i}_{w}100ranking5000', j, 'logs', 'test_metrics.json'))
                    r += f" &  "
        if cc % (len(a4) * len(a1)) == 0:
            r += ' \\\\  \\bottomrule \n\\end{tabular}' 
        elif cc % len(a4) == 0:
            r += ' \\\\  \\midrule \n' 
        else:
            r += ' \\\\  \n'
        
print(r)