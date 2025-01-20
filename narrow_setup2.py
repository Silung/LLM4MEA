import os
import sys
import time
import platform
import re
import subprocess

cmd_list = []
for data_name in ['beauty', 'games', 'steam']:
    for arch1 in ['narm', 'sas', 'bert']:
        for arch2 in ['narm', 'sas', 'bert']:
            if arch1 != arch2:
                val = 'decay' if data_name == 'beauty' else 'top1'
                cmd_list.append(f'python distill.py --dataset_code {data_name} --model_code {arch1} --bb_model_code {arch2} --num_generated_seqs 5001 --generated_sampler llm_seq --batch_size 512 --id 51 --few_shot 0 --num_epochs 300 --val_strategy {val} > temp/{data_name}_model_code_{arch1}_bb_model_code_{arch2}.txt')

def gpu_info():
    system = platform.system()
    gpus_memory = []
    # gpus_power = []
    gpus_util = []
    all_gpu_status = os.popen('nvidia-smi | grep %').read().strip().split('\n')
    for gpu_status in all_gpu_status:
        gpu_memory = re.search(r'(\d+)MiB', gpu_status).group(1)
        gpu_util = re.search(r'(\d+)%', gpu_status).group(1)
        gpus_memory.append(int(gpu_memory))
        gpus_util.append(int(gpu_util))
        # gpus_power.append(int(gpu_status[1].split('   ')[-1].split('/')[0].split('W')[0].strip()))
    # return gpus_power, gpus_memory
    return gpus_memory, gpus_util


def narrow_setup(cmd, interval=2):
    while True:
        gpus_memorys, gpus_util = gpu_info()
        for idx, (gpu_memory, gpu_util) in enumerate(zip(gpus_memorys, gpus_util)):
            if gpu_util > 30 or gpu_memory > 6000 :  # set waiting condition
                # symbol = 'monitoring ' + '>' * idx + ' ' * (10 - i - 1) + '|'
                # gpu_power_str = 'gpu power%d W |' % gpu_power
                gpu_memory_str = 'gpu memory: %d MiB ' % gpu_memory
                sys.stdout.write(f'\r gpu{idx}: ' + gpu_memory_str + f"gpu util: {gpu_util}% ")
                time.sleep(2)
                sys.stdout.flush()
                pass
            else:
                cuda_cmd = f"export CUDA_VISIBLE_DEVICES={idx} && {cmd}"
                print(f'\nexec: {cuda_cmd}')
                process = subprocess.Popen(cuda_cmd, shell=True)
                time.sleep(interval)
                return process
        time.sleep(interval)


if __name__ == '__main__':
    for cmd in cmd_list:
        narrow_setup(cmd, 60)
        time.sleep(30)
