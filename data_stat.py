import pickle
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

with open('data/preprocessed/ml-1m_text.pkl', 'rb') as f:
    text = pickle.load(f)
item_info = text['item']

most_common10 = None

def stat(path):
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    seqs = dataset['seqs']

    # titles = np.vectorize(item_info['title'].get)(seqs)
    genres = np.vectorize(item_info['genres'].get)(seqs)
    
    data = Counter(genres.reshape(-1))
    return data


data1 = stat('gen_data/ml-1m/bert_2500_10/autoregressive_dataset.pkl')
data2 = stat('gen_data/ml-1m/bert_2500_10/llm_dataset.pkl')

data_sum = (data1 + data2).most_common()[:10]
data_sum_keys = dict(data_sum).keys()

data1 = {k:dict(data1)[k] for k in data_sum_keys}
data2 = {k:dict(data2)[k] for k in data_sum_keys}
y1 = data1.values()
y2 = data2.values()

print(f'ag data:\n {data1}')
print(f'llm data:\n {data2}')



plt.figure(figsize = (10,8))
plt.bar(np.arange(len(y1)), y1, width=0.4, color='tomato', label='AG')
# for x, y in enumerate(y1):
#     plt.text(x, y, y)
    
plt.bar(np.arange(len(y2)) + 0.4, y2, width=0.4, color='steelblue', label='LLM')
# for x, y in enumerate(y2):
#     plt.text(x + 0.4, y, y)

br_data_sum_keys = ['\n'.join(i.split('|')) for i in data_sum_keys]
plt.xticks(np.arange(len(data_sum_keys)) + 0.5 / 2, br_data_sum_keys)
plt.xticks(rotation=45)
plt.xticks(fontsize = 8)

# 设置标题和轴标签
plt.title('Counts')
plt.xlabel('genres')
plt.ylabel('Count')
plt.savefig('data_top10.jpg')