import os
import json
import numpy as np

id = 33
a1 = ['narm', 'sas', 'bert']
# a2 = ['games', 'steam', 'beauty']
a2 = ['beauty', 'games', 'steam']
a3 = ['NDCG@10', 'Recall@10', 'Agr@1', 'Agr@10']
# a4 = ['target', 'self', 'random', 'autoregressive', f'llm_seq{id}']
a4 = [f'llm_seq']
a1map = {'narm':'NARM', 'sas':'SASRec', 'bert':'BERT'}
a4map = {'target':'Target', 'self':'Secret', 'random':'Random', 'autoregressive':'DFME', f'llm_seq':'Ours'}

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
r = ''
# r = r"""\begin{tabular}{lccccccccccccc}
# \toprule
#  & & \multicolumn{4}{c}{ML-1M}   & \multicolumn{4}{c}{Steam} & \multicolumn{4}{c}{Beauty}  \\ 
#  \cmidrule(l){3-6} \cmidrule(l){7-10} \cmidrule(l){11-14}
#                 &&N@10 & R@10 & Agr@1 & Agr@10 & N@10 & R@10 & Agr@1 & Agr@10 & N@10 & R@10 & Agr@1 & Agr@10\\ \midrule
# """

cc = 0
for i in a1:
    r += f'\\multirow{{{len(a4)}}}{{*}}{{{a1map[i]}}} '
    for w in a4:
        cc += 1
        r += f'& {a4map[w]} '
        for j in a2:
            for k in a3:
                try:
                    if w == 'target':
                        path = os.path.join('experiments', i, j, 'logs')
                        files = [f for f in os.listdir(path) if f.startswith('test_metrics_') and f.endswith('.json')]
                        if len(files) != 1:
                            pass
                        else:
                            f = open(os.path.join(path, files[0]), 'r')
                        json_str = "".join(f.readlines())
                        data = json.loads(json_str)
                        if k in data:
                            r += f" & {data[k]:.4f}"
                        else:
                            r += f" & - "
                    else:
                        path = os.path.join('experiments', 'distillation_rank', f'{i}2{i}_{w}100ranking5001', j, 'logs')
                        # files = [f for f in os.listdir(path) if f.startswith('test_metrics_') and f.endswith('.json')]
                        files = [f for f in os.listdir(path) if f.startswith(f'test_metrics_{id}') and f.endswith('.json')]
                        print(f'len(files): {len(files)}')
                        metrics = []
                        for file in files:
                            with open(os.path.join(path, file), 'r') as f:
                                data = json.load(f)
                                metrics.append(data[k])
                        # print(len(metrics))
                        metrics = np.array(metrics)

                        # f = open(os.path.join('experiments', 'distillation_rank', f'{i}2{i}_{w}100ranking5000', j, 'logs', 'test_metrics.json'), 'r')    
                        # json_str = "".join(f.readlines())
                        # data = json.loads(json_str)
                        # print(f"{i}: \t{data[i]}\t", end='')
                        # r += f" & {np.mean(metrics):.4f}$\pm${np.std(metrics):.4f}"
                        r += f" & {np.mean(metrics):.4f}"
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