import math

for j in range(3):
    dataset_name = ['beauty', 'games', 'steam'][j]
    data_info = {'beauty':54542, 'games':23464, 'steam':13046}
    k = 0
    m = data_info[dataset_name]
    n = int(0.9*m)
    for i in range(m-n+1,m+1):
        k += 1/i
    print(f'{dataset_name}\n k: {math.ceil(m*k)}\n seq: {math.ceil(m*k/50)}')