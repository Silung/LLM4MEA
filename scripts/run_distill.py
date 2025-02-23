import os
import subprocess
import time
import pickle


def rename_old(path):
    if os.path.exists(path):
        target_path = path+'.old'
        rename_old(target_path)
        os.rename(path, target_path)
    else:
        return


# 配置
dis_gpu = [2]  # 可用显卡
seqs_per_proc = 500
id_start = 0
num_p = 50 - id_start
arch = 'narm'
dataset_name = ['ml-1m', 'steam', 'beauty'][0]
sampler = ['llm_seq', 'random', 'autoregressive', 'self'][1]
loss = ['ranking', 'ce+ranking'][0]
total_seqs = seqs_per_proc * num_p    # num_generated_seqs的总和

step = 10

# llama 和 distill 程序路径
distill_dir = "/data/zhaoshilong/REA_with_llm"

root = f'gen_data/{dataset_name}/{arch}_{seqs_per_proc}_100'


# 所有进程完成后，运行最后的任务
processes = []
# for idx in range(num_p // step):
for idx in range(1,5):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(dis_gpu[idx%len(dis_gpu)])
    final_cmd = f"python {distill_dir}/distill.py --dataset_code {dataset_name} --model_code {arch} --bb_model_code {arch} " \
                f"--num_generated_seqs {step*seqs_per_proc} --generated_sampler {sampler} -k 100 --id {idx} --loss {loss}"
    print(f"启动最终 distill 任务：{final_cmd}")
    p = subprocess.Popen(final_cmd, shell=True, cwd=distill_dir)
    processes.append(p)
    time.sleep(60)

# 等待所有子进程完成
for p in processes:
    p.wait()
    
# for idx in range(num_p // step):
#     os.environ['CUDA_VISIBLE_DEVICES'] = str(dis_gpu[idx%len(dis_gpu)])
#     final_cmd = f"python {distill_dir}/distill.py --dataset_code {dataset_name} --model_code {arch} --bb_model_code {arch} " \
#                 f"--num_generated_seqs {step*seqs_per_proc} --generated_sampler {sampler} -k 100 --id {idx}"
#     print(f"启动最终 distill 任务：{final_cmd}")
#     subprocess.run(final_cmd, shell=True, cwd=distill_dir)