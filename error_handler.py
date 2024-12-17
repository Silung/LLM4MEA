from openai import AzureOpenAI
import os 
import json
import time

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-10-21",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

response = client.batches.retrieve('batch_14b7a8c5-8f54-4626-91af-ebe79724ca25')
file_response = client.files.content(response.output_file_id)

json_lines = file_response.text.strip().split('\n')
loaded_data = [json.loads(line) for line in json_lines]
id2msg = {data['custom_id']:data['response']['body']['choices'][0]['message'] for data in loaded_data}

with open('batch_log/batch_input_steam_bert_5.jsonl', 'r') as f:
    t = f.read().strip().split('\n')
    msg = [json.loads(line)["body"]["messages"] for line in t]

# 不好的结果，重试
for i, (k, v) in enumerate(id2msg.items()):
    cc_try = 0
    while 'content' not in v and cc_try < 6:
        print(v)
        cc_try += 1
        m = msg[int(k.split('-')[-1])]
        while True:
            try:
                chat_response = client.chat.completions.create(model="gpt-4o-mini-chat", messages=m)
                break
            except:
                print(f"Update {k} failed.")
                time.sleep(60)
        v = dict(chat_response.choices[0].message)
        print(f"Update {k}.")
        time.sleep(15)
    if cc_try == 6:
        print(msg[int(k.split('-')[-1])])
        print("Answer:")
        v = {'content': str(input()), 'role': 'assistant'}
    id2msg[k] = v
    loaded_data[i]['response']['body']['choices'][0]['message'] = v
    json_lines[i] = json.dumps(loaded_data[i])

batch_output_path = os.path.join('batch_log', f'batch_output_steam_bert_8.txt')
with open(batch_output_path, 'w') as f:
    # f.write(file_response.text)
    f.write('\n'.join(json_lines))