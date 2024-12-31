import pickle
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

most_common10 = None

def stat(path, item_info):
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    seqs = dataset['seqs']

    # print('8891' in item_info['genres'])
    item_info['genres'] = {int(k):v for k,v in item_info['genres'].items()}
    genres = np.vectorize(item_info['genres'].get)(seqs)

    cc = []
    for genre in genres:
        # if len(set(genre)) != 1:
        #     print(genre)
        cc.append(len(set(genre)))
        
    data = Counter(cc)
    return data

def stat_org(path, item_info):
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    
    smap = dataset['smap']
    r_smap = {v: k for k, v in smap.items()}
    cc = []
    for k, v in dataset['train'].items():
        org_items = np.vectorize(r_smap.get)(v)
        genre = np.vectorize(item_info['genres'].get)(org_items)
        cc.append(len(set(list(genre))))
    
    data = Counter(cc)
    return data

def diversity(name, path):
    if 'ml-1m' in path:
        with open('data/preprocessed/ml-1m_text.pkl', 'rb') as f:
            text = pickle.load(f)
        item_info = text['item']
    elif 'steam' in path:
        with open('data/preprocessed/steam_text.pkl', 'rb') as f:
            text = pickle.load(f)
        item_info = text['item']
    else:
        raise("Not Implemented")


    try:
        data = stat(path, item_info)
    except:
        data = stat_org(path, item_info)

    data_keys = sorted(list(dict(data).keys()))

    data = {k:dict(data)[k] for k in data_keys}
    y = data.values()

    print(f'{name} data:\n {data}')


    plt.figure(figsize = (10,8))
    plt.bar(np.arange(len(y)), y, width=0.4, color='tomato', label='AG')
    # for x, y in enumerate(y1):
    #     plt.text(x, y, y)
        
    # plt.bar(np.arange(len(y2)) + 0.4, y2, width=0.4, color='steelblue', label='LLM')
    # for x, y in enumerate(y2):
    #     plt.text(x + 0.4, y, y)

    # br_data_sum_keys = ['\n'.join(i.split('|')) for i in data_sum_keys]
    plt.xticks(np.arange(len(data_keys)), data_keys)
    # plt.xticks(rotation=45)
    plt.xticks(fontsize = 8)

    # 设置标题和轴标签
    plt.title(f'Diversity of Data from {name}')
    plt.xlabel('Diversity')
    plt.ylabel('Count')
    plt.savefig(f'pics/data_diversity_{name}.jpg')

def diversity_autoregressive(name, path):
    cc = []
    all_items = set()
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    candidates = dataset['candidates']
    # 计算所有candidates的平均值
    cc_all = []
    for i in range(len(candidates)):
        cc_i = []
        all_items_i = set()
        for candidate in candidates[i]:
            div = set(candidate)
            cc_i.append(len(div - all_items_i))
            all_items_i = all_items_i | div
        cc_all.append(cc_i)
    
    # 计算每个位置的平均值
    max_len = max(len(x) for x in cc_all)
    cc = []
    for i in range(max_len):
        values = [x[i] for x in cc_all if i < len(x)]
        cc.append(sum(values) / len(values))
    
    return cc

def diversity_random_baseline(name):
    # num_items = 54542  # beauty数据集的item数量
    num_items = 13046  # steam数据集的item数量
    target_len = 50
    num_samples = 5000
    
    cc_all = []
    for _ in range(num_samples):
        cc_i = []
        all_items_i = set()
        for _ in range(target_len):
            # 每次随机采样100个item
            sample = set(np.random.randint(1, num_items+1, size=100))
            # 计算新出现的item数量
            cc_i.append(len(sample - all_items_i))
            all_items_i = all_items_i | sample
        cc_all.append(cc_i)
    
    # 计算每个位置的平均值
    cc = []
    for i in range(target_len):
        values = [x[i] for x in cc_all]
        cc.append(sum(values) / len(values))
    
    return cc

# diversity('narm', 'gen_data/ml-1m/narm_5000_100/llm_seq_dataset.pkl.old')
# diversity('narm-new', 'gen_data/ml-1m/narm_5000_100/llm_seq_dataset.pkl')
# diversity('narm-ag', 'gen_data/ml-1m/narm_5000_100/autoregressive_dataset.pkl')
# diversity('sas', 'gen_data/ml-1m/sas_5000_100/llm_seq_test_dataset.pkl')
# diversity('sas-ag', 'gen_data/ml-1m/sas_5000_100/autoregressive_dataset.pkl')
# diversity('ml-1m', 'data/preprocessed/ml-1m_min_rating0-min_uc5-min_sc5-splitleave_one_out/dataset.pkl')
# diversity('bert', 'gen_data/ml-1m/bert_5000_100/llm_seq_dataset.pkl')
# diversity('bert-exam', 'gen_data/ml-1m/bert_5000_100/llm_exam_dataset.pkl')
# diversity('bert-random', 'gen_data/ml-1m/bert_5000_100/random_dataset.pkl')
# diversity('narm-org', 'gen_data/steam/narm_5000_100/llm_seq_dataset.pkl')
# diversity('narm-ml-len=50', 'gen_data/ml-1m/narm_5000_100/llm_seq_dataset.pkl')
# diversity_autoregressive('bert_5001_100', 'gen_data/beauty/bert_5001_100/autoregressive0_dataset.pkl')

plt.figure(figsize=(10,8))

cc_random = diversity_autoregressive('llm_seq', 'gen_data/steam/bert_5001_100/llm_seq50_dataset.pkl')
cc_auto = diversity_autoregressive('autoregressive', 'gen_data/steam/bert_5001_100/autoregressive0_dataset.pkl')
# cc_baseline = diversity_random_baseline('baseline')

plt.plot(range(len(cc_random)), cc_random, color='steelblue', label='llm_seq')
plt.plot(range(len(cc_auto)), cc_auto, color='tomato', label='Autoregressive')
# plt.plot(range(len(cc_baseline)), cc_baseline, color='green', label='Baseline')

plt.title('Diversity Comparison')
plt.xlabel('Sequence Index')
plt.ylabel('Diversity Count')
plt.legend()
plt.savefig('pics/data_diversity_comparison.jpg')
