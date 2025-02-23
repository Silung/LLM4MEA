import json
f = open('/data/zhaoshilong/REA_with_llm/batch_log/batch_output_steam_bert_11.txt','r')
text = f.read()

json_lines = text.strip().split('\n')
for i in range(5001):
    try:
        json.loads(json_lines[i])
    except:
        print(i)
        print(json_lines[i])