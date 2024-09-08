import os
import subprocess
import time
import pickle

# 配置
gpus = [1,2,4,5,6]  # 可用显卡列表
total_seqs = 5000    # num_generated_seqs的总和
id_start = 20
dataset_name = 'steam'
arch = 'narm'

llama_port_start = 1965
distill_port_start = 2000
master_port_start = 19605

# llama 和 distill 程序路径
llama_dir = "/data/zhaoshilong/llama3"
distill_dir = "/data/zhaoshilong/REA_with_llm"

# 计算每个程序生成的序列数量 (均分，不考虑剩余)
seqs_per_proc = total_seqs // len(gpus)
if total_seqs % len(gpus) != 0:
    seqs_per_proc += 1

# 记录所有子进程
processes = []

# 启动 Llama 服务器
for i, gpu in enumerate(gpus):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    llama_port = llama_port_start + i
    master_port = master_port_start + i
    cmd = f"conda run -n llama torchrun --master_port {master_port} --nproc_per_node 1 {llama_dir}/llama_chat_server.py " \
          f"--ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model " \
          f"--max_seq_len 10000 --max_batch_size 1 --port {llama_port} --temperature 1 --top_p 0.95"
    print(f"启动 Llama 服务器：{cmd}")
    p = subprocess.Popen(cmd, shell=True, cwd=llama_dir)
    # processes.append(p)
time.sleep(20)  # 给每个程序一些时间来启动

# 启动 distill.py 程序
for i, gpu in enumerate(gpus):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    distill_port = distill_port_start + i
    distill_cmd = f"conda run -n rea python {distill_dir}/distill.py --dataset_code {dataset_name} --model_code {arch} --bb_model_code {arch} " \
                  f"--num_generated_seqs {seqs_per_proc} --generated_sampler llm_seq -k 100 --port {llama_port_start + i} " \
                  f"--id {id_start + i} --gen_data_only --few_shot --device cpu"
    print(f"启动 distill 程序：{distill_cmd}")
    p = subprocess.Popen(distill_cmd, shell=True, cwd=distill_dir)
    processes.append(p)

# 等待所有子进程完成
for p in processes:
    p.wait()


root = f'gen_data/{dataset_name}/{arch}_{seqs_per_proc}_100'
paths = [f'llm_seq{i}_dataset.pkl' for i in range(id_start, id_start+len(gpus))]
print(paths)
datasets = []
for path in paths:
    with open(os.path.join(root, path), 'rb') as f:
         datasets.append(pickle.load(f))
    
seqs, logits, candidates = [], [], []
for dataset in datasets:
    seqs += dataset['seqs']
    logits += dataset['logits']
    candidates += dataset['candidates']
dataset3 = {'seqs':seqs[:total_seqs], 'logits':logits[:total_seqs], 'candidates':candidates[:total_seqs]}
print(f'len(dataset3):{len(dataset3)}')
print(f'len(dataset3["seqs"]):{len(dataset3["seqs"])}')

def rename_old(path):
    if os.path.exists(path):
        target_path = path+'.old'
        rename_old(target_path)
        os.rename(path, target_path)
    else:
        return

if not os.path.exists(f'gen_data/{dataset_name}/{arch}_{total_seqs}_100'):
    os.mkdir(f'gen_data/{dataset_name}/{arch}_{total_seqs}_100')
target_path = os.path.join(f'gen_data/{dataset_name}/{arch}_{total_seqs}_100', 'llm_seq_dataset.pkl')
rename_old(target_path)
with open(target_path, 'wb') as f:
    pickle.dump(dataset3, f)


# 所有进程完成后，运行最后一个任务
final_cmd = f"python {distill_dir}/distill.py --dataset_code {dataset_name} --model_code {arch} --bb_model_code {arch} " \
            f"--num_generated_seqs {total_seqs} --generated_sampler llm_seq -k 100"
print(f"启动最终 distill 任务：{final_cmd}")
subprocess.run(final_cmd, shell=True, cwd=distill_dir)
