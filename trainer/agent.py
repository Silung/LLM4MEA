import re
import os
import ast
import pickle
import socket
import json
import torch
import time
import random
import numpy as np
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from tqdm import trange
from datetime import datetime


from datasets import dataset_factory


class Agent():
    def __init__(self, args):
        self.args = args
        self.dataset_code = args.dataset_code
        dataset = dataset_factory(args)
        data = dataset.load_dataset()
        self.umap = data['umap']
        self.smap = data['smap']
        self.r_umap = {v: k for k, v in self.umap.items()}
        self.r_smap = {v: k for k, v in self.smap.items()}
        
        text = dataset.load_text()
        self.user_info = text['user']
        self.item_info = text['item']
        
        self.mem = None
        self.watched_items = None
        self.t = 70
        self.call_cc = 0
        # while os.path.exists(f'batch_log/batch_input_steam_narm_{self.call_cc + 1}.jsonl') and not self.args.debug:
        #     self.call_cc += 1
        
        # self.init_profiles(self.args.num_generated_seqs)
        try:
            with open('examples_in_ml.txt', 'r') as f:
                self.history = f.read().split('\n')
        except:
            self.history = None
        
        if self.dataset_code == 'ml-1m':
            self.his_size = 20
            self.seq_size = 10
        elif self.dataset_code == 'steam':
            self.his_size = 20
            self.seq_size = 5
        else:
            self.his_size = 10
            self.seq_size = 5
            
    def set_history(self, seqs):
        org_items = np.vectorize(self.r_smap.get)(seqs.cpu()) 
        titles = np.vectorize(self.item_info['title'].get)(org_items)
        self.watched_items = titles.tolist()
        
    def send_rev(self, msg):
        self.call_cc += 1
        if self.args.llm == 'llama':
            # 创建TCP socket
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            # 连接服务器
            client_socket.connect(('localhost', self.args.port))
            
            # 创建字典
            data_to_send = msg
            
            # 序列化字典为JSON格式
            data_to_send_json = json.dumps(data_to_send)
            
            # print(data_to_send_json)
            # 发送数据
            client_socket.send(data_to_send_json.encode())
            # 接收数据
            received_data = client_socket.recv(40960).decode()
            
            print(received_data)
            
            # 关闭连接
            client_socket.close()
            
            received_data = json.loads(received_data)
        elif self.args.llm == 'gpt-4o-mini_batch':
            from openai import OpenAI
            
            if not os.path.exists('batch_log'):
                os.makedirs('batch_log')
            batch_input_path = os.path.join('batch_log', f'batch_input_{self.args.dataset_code}_{self.args.bb_model_code}_{self.call_cc}.jsonl')
            batch_output_path = os.path.join('batch_log', f'batch_output_{self.args.dataset_code}_{self.args.bb_model_code}_{self.call_cc}.txt')
            
            if os.path.exists(batch_output_path):
                with open(batch_output_path, 'r') as f:
                    text = f.read()
                json_lines = text.strip().split('\n')                    
            else:
                if not os.path.exists(batch_input_path):
                    with open(batch_input_path, 'w') as f:
                        for idx, m in enumerate(msg):
                            d = {"custom_id": f"{self.args.dataset_code}-{self.args.bb_model_code}-{idx}", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o-mini", "messages": m, "max_tokens": 5000}}
                            # print(d)
                            f.write(json.dumps(d) + '\n')
                
                client = OpenAI()
                batch_input_file = client.files.create(file=open(batch_input_path, "rb"), purpose="batch")
                batch_input_file_id = batch_input_file.id

                batch_info = client.batches.create(
                    input_file_id=batch_input_file_id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={
                    "description": f"{self.args.dataset_code}-{self.args.bb_model_code}"
                    }
                    )
                time.sleep(10)

                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"Submit at {current_time}\t Submit batch task: {self.args.dataset_code}-{self.args.bb_model_code}-{batch_info.id}")

                # batch_input_file_id = 'batch_6704f47a53948190bd447a82ec4b7e64'
                while True:
                    try:
                        response = client.batches.retrieve(batch_info.id)
                        if response.status == 'completed':
                            break
                        elif response.status in ['failed', 'expired', 'cancelling', 'cancelled']:
                            print(f'API error code: {response.status}')
                            raise NotImplementedError()
                    except Exception as e:
                        print(f'Network Error: {str(e)}')
                    time.sleep(2 * 60)
                time.sleep(10)
                
                file_response = client.files.content(response.output_file_id)
                batch_output_path = os.path.join('batch_log', f'batch_output_{self.args.dataset_code}_{self.args.bb_model_code}_{self.call_cc}.txt')
                with open(batch_output_path, 'w') as f:
                    f.write(file_response.text)
                json_lines = file_response.text.strip().split('\n')

            loaded_data = [json.loads(line) for line in json_lines]
            id2msg = {data['custom_id']:data['response']['body']['choices'][0]['message'] for data in loaded_data}
            e = set([f"{self.args.dataset_code}-{self.args.bb_model_code}-{idx}" for idx in range(len(msg))]) - set(id2msg.keys())
            if len(e) > 0:
                print(f'Fails id: {list(e)}')
            received_data = [id2msg[f"{self.args.dataset_code}-{self.args.bb_model_code}-{idx}"] for idx in range(len(msg))]
            # received_data = [id2msg[idx] for idx in ['request-1', 'request-2']]
        
        elif self.args.llm == 'gpt-test':
            batch_output_path = os.path.join('batch_log.old', f'batch_output_steam_narm_0.txt')
            with open(batch_output_path, 'r') as f:
                text = f.read()
            json_lines = text.strip().split('\n')
            loaded_data = [json.loads(line) for line in json_lines]
            id2msg = {data['custom_id']:data['response']['body']['choices'][0]['message'] for data in loaded_data}
            received_data = [id2msg[f"{self.args.dataset_code}-{self.args.bb_model_code}-{idx}"] for idx in range(len(msg))]
        # print(received_data)
        return received_data
    
    def encode(self, sorted_items):
        # input: converted item id list, start with 1
        # output: list of msg, msg is a list of ditc, a dict likes that: {"role": "user", "content": "XXX"}
        # output = [[
        #     {"role": "user", "content": "XXX"},
        # ]]
        # print(type(sorted_items))
        # print(sorted_items)
        
        self.org_items = np.vectorize(self.r_smap.get)(sorted_items)
        self.titles = np.vectorize(self.item_info['title'].get)(self.org_items)
        if self.dataset_code in ['ml-1m', 'steam']:
            self.genres = np.vectorize(self.item_info['genres'].get)(self.org_items)
        elif self.dataset_code == 'beauty':
            self.genres = None
        elif self.dataset_code == 'games':
            self.genres = None
        # output = self.template1(self.titles, self.genres)
        output = self.template(self.titles, self.genres)

        return output
    
    def find(self, t_list, s):
        t_list =list(t_list)
        best_match = process.extractOne(s, t_list)
        if best_match[1] > self.t:
            idx = t_list.index(best_match[0])
        else:
            idx = random.randint(0, len(t_list)-1)
        return idx
    
    def decode(self, received_data):
        # input: llm return
        # output: index in sorted_items
            
        text = [ans['generation']['content'] for ans in received_data]

        # selected_indices = np.vectorize(find, signature='(n),(n)->(n)')(self.titles, text)
        selected_indices = [self.find(self.titles[i], text[i]) for i in range(len(text))]
        [self.watched_items[i].append(self.titles[i][selected_indices[i]]) for i in range(len(text))]
        
        for i, ans in enumerate(received_data):
            self.mem[i].append(ans['generation'])
        
        return selected_indices
        
    def template1(self, titles, genres):
        # output: list shaped same as Batch
        
        def join(ts, gs):
            t = ''
            for i in range(len(ts)):
                t += f'{i+1}. {ts[i]}({gs[i]}); '
            return t
        
        texts = np.vectorize(join, signature='(n),(n)->()')(titles, genres)
        
        if self.watched_items is None:
            sys = {"role": "system", "content": "You're a user with your own hobbies. Now, I want to recommend a few more movies to you. Select one that you are most interested in. Only answer the title!"}
            output = [[sys, {"role": "user", "content": text}] for text in texts]
        else:
            output = [[{"role": "user", "content": text}] for text in texts]
            # for i, text in enumerate(texts):
            #     self.mem[i] = self.mem[i][:1] + self.mem[i][-4:]
            #     self.mem[i].append({"role": "user", "content": text}) 
        # output = self.mem.copy()

        if self.mem is None:
            self.mem = output
        else:
            for i, text in enumerate(output):
                self.mem[i] = self.mem[i][:1] + self.mem[i][-4:]
                self.mem[i] += text

        return output

    def template(self, titles, genres):
        # output: list shaped same as Batch
        
        def join(ts, gs):
            t = ''
            for i in range(len(ts)):
                if self.dataset_code in ['ml-1m', 'steam']:
                    t += f'{i+1}. {ts[i]}({gs[i]}); '
                elif self.dataset_code == 'beauty':
                    t += f'{i+1}. {ts[i]}; '
            return t
        
        texts = np.vectorize(join, signature='(n),(n)->()')(titles, genres)
        
        if self.watched_items is None:
            # sys = {"role": "system", "content": ""}
            if self.dataset_code == 'ml-1m':
                output = [[{"role": "user", "content": f'You are a user with your own preference. Now, I want to recommend a few more movies to you. Select one that you are most interested in. Only answer the title! {text}'}] for text in texts]
            elif self.dataset_code in ['beauty', 'steam']:
                output = [[{"role": "user", "content": f'You are a user with your own preference. Now, I want to recommend a few more items to you. Select one that you are most interested in. Only answer the title! {text}'}] for text in texts]
        else:
            output = []
            for i, text in enumerate(texts):
                # print(self.mem[i])
                ex_items = ', '.join(self.watched_items[i][-self.his_size:])
                if self.dataset_code == 'ml-1m':
                    output.append([{"role": "user", "content": f"You are a user with your own preference, and you have saw {ex_items}. Now, I want to recommend a few more movies to you. Choose a movie that you are interested in and have not watched yet. Only answer the title without other words! {text}"}])
                elif self.dataset_code in ['beauty', 'steam']:
                   output.append([{"role": "user", "content": f"You are a user with your own preference, and you have bought {ex_items}. Now, I want to recommend a few more items to you. Choose an items that you are interested in and have not selected yet. Only answer the title without other words! {text}"}])
                
        if self.mem is None:
            self.mem = [[] for i in range(len(titles))]
        if self.watched_items is None:
            self.watched_items = [[] for i in range(len(titles))]
        return output
    
    def clear(self):
        self.mem = None    
    
    def __call__(self, sorted_items):
        sorted_items = np.array(sorted_items)
        query = self.encode(sorted_items)
        received_data = self.send_rev(query)
        if self.args.debug:
            print('LLM input and output:')
            print(query[0])
            print(received_data[0])
            print()
        selected_indices = self.decode(received_data)
        # shape: [B], index start with 1.
        # print(selected_indices[0])
        return selected_indices
    
    def init_profiles(self, size):
        pass

    def update_profiles(self, idxs, batch_size):
        pass
    
class ExampleAgent(Agent):
    def template(self, titles, genres):
        # output: list shaped same as Batch
        
        def join(ts, gs):
            t = ''
            for i in range(len(ts)):
                t += f'{i+1}. {ts[i]}({gs[i]}); '
            return t
        
        texts = np.vectorize(join, signature='(n),(n)->()')(titles, genres)
        
        j = random.randint(0, 9)
        if self.watched_items is None:
            # sys = {"role": "system", "content": ""}
            output = [[{"role": "user", "content": f'This is the viewing history of a certain user: {self.history[j]}. You are a user with your own hobbies. Now, I want to recommend a few more movies to you: {text}. Select one that you are most interested in. Only answer the title! '}] for text in texts]
        else:
            output = []
            for i, text in enumerate(texts):
                # print(self.mem[i])
                ex_items = ', '.join(self.watched_items[i][-self.his_size:])
                output.append([{"role": "user", "content": f"This is the viewing history of a certain user: {self.history[j]}. You are a user with your own hobbies, and you have saw {ex_items}. Now, I want to recommend a few more movies to you: {text}. Choose a movie that you are interested in and have not watched yet. Only answer the title without other words! "}])
                
        if self.mem is None:
            self.mem = [[] for i in range(len(titles))]
        if self.watched_items is None:
            self.watched_items = [[] for i in range(len(titles))]
        # print(output)
        return output
    
class SeqAgent(Agent):
    def template(self, titles, genres):
        # output: list shaped same as Batch
        
        def join1(ts):
            # t = ''
            # for i in range(len(ts)):
            #     t += f'{i+1}. {ts[i]}({gs[i]}); '
            ts = [i[:100] for i in ts]
            t = ', '.join(ts)
            return t
        
        def join2(ts):
            # t = ''
            # for i in range(len(ts)):
            #     t += f'{i+1}. {ts[i][:80]}; '
            ts = [i[:100] for i in ts]
            t = '| '.join(ts)
            return t
        
        if self.dataset_code in ['ml-1m', 'steam']:
            # texts = np.vectorize(join1, signature='(n),(n)->()')(titles, genres)
            texts = np.vectorize(join1, signature='(n)->()')(titles)
        elif self.dataset_code in ['beauty']:
            texts = np.vectorize(join2, signature='(n)->()')(titles)
        
        try:
            if self.args.cases_path is not None:
                with open(self.args.cases_path,'r') as f:
                    few_shot_cases = f.read()                
                    cases_txt = f"Cases: {few_shot_cases} "
            else:
                cases_txt = ""
        except:
            print("Few shot cases not exist.")
            cases_txt = ""
        
        # j = random.randint(0, 9)
        if self.watched_items is None:
            # sys = {"role": "system", "content": ""}
            # output = [[{"role": "user", "content": f'This is the viewing history of a certain user: {self.history[j]}. You are a user with your own hobbies. Now, I want to recommend a few more movies to you: {text}. Select 10 movies from them, then return the titles in the order you plan to watch in the format of python list. Make sure only answer the title without other words! '}] for text in texts]
            if self.dataset_code == 'ml-1m':
                output = [[{"role": "user", "content": f"I want to recommend a few more movies to you: {text}. Select {self.seq_size} movies from them, then return the titles in the order you plan to watch in the format of python list (i.e. ['a', 'b', 'c']). Make sure only answer the title without other words! Note that one user only accesses a few specific categories of items."}] for text in texts]
            elif self.dataset_code == 'beauty':
                output = [[{"role": "user", "content": f"You are a user with your own preference. Now, I want to recommend a few more items to you: {text}. Select {self.seq_size} items from them, then return the titles in the order you plan to buy in the format of python list(i.e. ['a', 'b', 'c']). Make sure only answer the title without other words! "}] for text in texts]
            elif self.dataset_code == 'steam':
                # last update
                # output = [[{"role":"system", "content":"You are a user of the Steam, a video game digital distribution service and storefront, and want to purchase your next game."}, 
                #             {"role": "user", "content": f"I want to recommend a few more items to you: {text}. Select {self.seq_size} items from them, then return the titles in the order you plan to buy in the format of python list (i.e. ['a', 'b', 'c']). Make sure only answer the title without other words! Note that one user only accesses a few specific categories of items."}] for text in texts]
                # output = [[{"role":"system", "content":"You are a user on Steam, a digital distribution service for video games, looking to purchase your next game."},
                #             {"role": "user", "content": f"Based on your gaming preferences and past interactions, I have a list of game candidates for you: {text}. Popular game genres include Action, Adventure, Role-Playing (RPG), Strategy, Simulation, and Sports. Please select {self.seq_size} games from this list that you would consider purchasing next, focusing on the genres you usually enjoy. However, keep in mind that users sometimes explore games outside of their typical preferences. Return only the selected game titles in the Python list format (e.g., ['title1', 'title2', 'title3']). Make sure to respond with only the titles and nothing else!"}] for text in texts]
                # output = [[{"role": "user", "content": f"You are a user with your own preference. Now, I want to recommend a few more items to you: {text}. Select {self.seq_size} items from them, then return the titles in the order you plan to buy in the format of python list(i.e. ['a', 'b', 'c']). Make sure only answer the title without other words! "}] for text in texts]
                
                # output = [[{"role":"system", "content":"You are a user on Steam, a digital distribution service for video games."},
                #             {"role": "user", "content": f"Based on your gaming preferences and past interactions, I have a list of game candidates for you. Please select {self.seq_size} games from this list that you are most likely to purchase. Return only the selected game titles in the Python list format with pure text (e.g., ['title1', 'title2']). Make sure to respond with only the titles and nothing else! {cases_txt}Now, recommendations for you: {text}."}] for text in texts]
                output = [[{"role": "user", "content": f"Based on your preferences, select {self.seq_size} games from the list below. Return only the selected game titles in the Python list format with pure text (e.g., ['title1', 'title2']). Respond with only the titles! {cases_txt}Here are your recommendations: {text}."}] for text in texts]
        else:
            output = []
            for i, text in enumerate(texts):
                # print(self.mem[i])
                # if len(', '.join(self.watched_items[i][-5:])) > 380:
                #     self.watched_items[i] = self.watched_items[i][:-self.seq_size]
                ex_len = min(self.his_size, len(self.watched_items[0]))
                if self.dataset_code in ['ml-1m', 'steam']:
                    sep = ', '
                else:
                    sep = '| '
                ex_items_head = sep.join(self.watched_items[i][:ex_len//2])
                ex_items_tail = sep.join(self.watched_items[i][ex_len//2:])
                # print(len(', '.join(self.watched_items[i][-5:])))
                if self.dataset_code == 'ml-1m':
                    # output.append([{"role": "user", "content": f'This is the viewing history of a certain user: {self.history[j]}. You are a user with your own hobbies, and you have saw {ex_items}. Now, I want to recommend a few more movies to you: {text}. Select 10 movies from them, then return the titles in the order you plan to watch in the format of python list. Make sure not to choose movies you have watched and only answer the title without other words! '}])
                    # output.append([{"role": "user", "content": f'You have saw {ex_items_head}...{ex_items_tail}. I want to recommend a few more movies to you: {text}. Select {self.seq_size} movies from them, then return the titles in the order you plan to watch in the format of python list. Make sure not to choose movies you have watched and only answer the title without other words!  Note that one user only accesses a few specific categories of items, and please follow the preferences reflected in history.'}])
                    output.append([{"role": "user", "content": f"{self.profile[i]} Based on these preferences, select {self.seq_size} movies from the list below. Return only the selected movie titles in the Python list format with pure text (e.g., ['title1', 'title2']). Respond with only the titles! {cases_txt}Previously watched movies: {ex_items_head}...{ex_items_tail}. Here are your recommendations: {text}."}])
                elif self.dataset_code == 'beauty':
                    output.append([{"role": "user", "content": f"{self.profile[i]} Based on these preferences, select {self.seq_size} items from the list below. Return only the selected items titles in the Python list format with pure text (e.g., ['title1', 'title2']). Respond with only the titles! {cases_txt}Previously watched items: {ex_items_head}...{ex_items_tail}. Here are your recommendations: {text}."}])
                elif self.dataset_code == 'steam':
                    # last update 
                    # output.append([{"role":"system", "content":"You are a user of the Steam, a video game digital distribution service and storefront, and want to purchase your next game."}, 
                    #                {"role": "user", "content": f"You have bought {ex_items}. I want to recommend a few more items to you: {text}. Select {self.seq_size} items from them, then return the titles in the order you plan to buy in the format of python list (i.e. ['a', 'b', 'c']). Make sure only answer the title without other words! Note that one user only accesses a few specific categories of items, and please follow the preferences reflected in history."}])
                    # output.append([{"role": "system", "content": "You are a user on Steam, a digital distribution service for video games, looking to purchase your next game."},
                    #                 {"role": "user", "content": f"You have previously purchased the following games: {ex_items}. Based on your purchase history, I have a list of additional game recommendations for you: {text}. Please select {self.seq_size} games from this list that match your interests, and provide the titles in the order you would buy them, formatted as a Python list (e.g., ['title1', 'title2', 'title3']). Only respond with the list of titles, without any extra text! While users often stick to specific categories, they sometimes explore games outside their typical preferences, so consider this when making your selections."}])
                    # output.appsend([{"role": "user", "content": f"You are a user with your own preference, and you have bought {ex_items}. Now, I want to recommend a few more items to you: {text}. Select {self.seq_size} items from them, then return the titles in the order you plan to buy in the format of python list(i.e. ['a', 'b', 'c']). Make sure not to choose items you have bought and only answer the title without other words! "}])
                    
                    # output.append([{"role": "system", "content": "You are a user on Steam, a digital distribution service for video games."},
                    #                 {"role": "user", "content": f"Based on your purchase history, I have a list of additional game recommendations for you. Please select {self.seq_size} games from this list that you are most likely to purchase. Provide the titles in the Python list format with pure text (e.g., ['title1', 'title2']). Only respond with the list of titles, without any extra text! {cases_txt}You have previously viewed the following games: {ex_items_head}...{ex_items_tail}. Recommendations for you: {text}"}])
                    output.append([{"role": "user", "content": f"{self.profile[i]} Based on these preferences, select {self.seq_size} games from the list below.  Return only the selected game titles in the Python list format with pure text (e.g., ['title1', 'title2']). Respond with only the titles! {cases_txt}Previously viewed games: {ex_items_head}...{ex_items_tail}. Here are your recommendations: {text}"}])
        if self.mem is None:
            self.mem = [[] for i in range(len(titles))]
        if self.watched_items is None:
            self.watched_items = [[] for i in range(len(titles))]
        # print(output)
        return output
    
    def find(self, t_list, s):
        t_list =list(t_list)
        idx = []
        pattern = r'\[.*?\]'
        match = re.search(pattern, s)
        if match:
            list_str = match.group()
            # print(list_str)
            for s in list_str.split("', '"):
                best_match = process.extractOne(s, t_list)
                if best_match[1] > self.t:
                    idx.append(t_list.index(best_match[0]))
                else:
                    idx.append(random.randint(0, len(t_list)-1))
        else:
            idx = [random.randint(0, len(t_list)-1) for i in range(self.seq_size)]
        # print(len(idx))
        if len(idx) != self.seq_size:
            idx = [random.randint(0, len(t_list)-1) for i in range(self.seq_size)]
        return idx
    
    def decode(self, received_data):
        # input: llm return
        # output: index in sorted_items
            
        if self.args.llm == 'llama':
            text = [ans['generation']['content'] for ans in received_data]
        elif 'gpt' in self.args.llm:
            text = [ans['content'] for ans in received_data]
        else:
            raise NotImplementedError()

        # selected_indices = np.vectorize(find, signature='(n),(n)->(n)')(self.titles, text)
        selected_indices = [self.find(self.titles[i], text[i]) for i in range(len(text))]
        for i in range(len(text)):
            # new_his = [item[:100] for item in list(self.titles[i][selected_indices[i]])]
            new_his = list(self.titles[i][selected_indices[i]])
            # print(new_his[0])
            self.watched_items[i] += new_his
        
        for i, ans in enumerate(received_data):
            if self.args.llm == 'llama':
                self.mem[i].append(ans['generation'])
            elif 'gpt' in self.args.llm:
                self.mem[i].append(ans)
        return selected_indices
    
    def init_profile(self, seqs):
        seqs = np.array(seqs.cpu())
        org_seqs = np.vectorize(self.r_smap.get)(seqs)
        seqs_titles = np.vectorize(self.item_info['title'].get)(org_seqs)
        
        def join(ts, dataset_name):
            ts = [i[:100] for i in ts]
            if dataset_name == 'beauty':
                t = '| '.join(ts)
                q = f'Previously viewed items of a user on Amazon: {t}. Describe the perference of this user in one sentence.'
            elif dataset_name == 'ml-1m':
                t = ', '.join(ts)
                q = f'Previously viewed movies of a user on Movielens: {t}. Describe the perference of this user in one sentence.'
            elif dataset_name == 'steam':
                t = ', '.join(ts)
                q = f'Previously viewed games of a user on STEAM: {t}. Describe the perference of this user in one sentence.'
            return q
        
        text = [join(seq, self.dataset_code) for seq in seqs_titles]
        query = [[{"role": "user", "content": q}] for q in text]
        
        received_data = self.send_rev(query)
        if self.args.debug:
            print('LLM input and output:')
            print(query[0])
            print(received_data[0])
            print()
        
        if self.args.llm == 'llama':
            self.profile = [ans['generation']['content'] for ans in received_data]
        elif 'gpt' in self.args.llm:
            self.profile = [ans['content'] for ans in received_data]
        else:
            raise NotImplementedError()
        

class ProfileAgent(Agent):
    def extract_json_from_text(self, text):
        # 定义匹配JSON对象的正则表达式
        json_pattern = r'\{[^{}]*\}'
        
        # 使用re.findall来查找所有符合正则表达式的JSON对象
        json_strings = re.findall(json_pattern, text)
        
        # 尝试解析找到的JSON字符串，并返回有效的JSON对象
        json_objects = []
        for json_str in json_strings:
            try:
                json_obj = json.loads(json_str)
                json_objects.append(json_obj)
            except json.JSONDecodeError:
                continue
        
        return json_objects
    
    # def init_profiles(self, size):
    #     self.profiles = []
    #     print('Initing profiles...')
    #     profile_query = {"role": "user", "content": f'The user profiles include gender, age, traits, career, interests, and behavioral features. The traits describe the user\'s personality, such as being "compassionate", "ambitious", or "optimistic". The interests indicate the user\'s preferences on the items, such as "sci-fi movies" or "comedy videos". Now generate a random user profile and only return the profile content with json format.'}
    #     query = None
    #     for i in trange(size):
    #         if query is None:
    #             query = [[profile_query]]
    #         else:
    #             query = [query[0][-26:]]
    #             query[0].append(profile_query)
    #         received_data = self.send_rev(query)
    #         query[0].append(received_data[0]['generation'])
    #         for data in received_data:
    #             jsons = self.extract_json_from_text(data['generation']['content'])
    #             if len(jsons) == 1:
    #                 self.profiles.append(jsons[0])
    #             else:
    #                 raise("Init user profiles error!")
    
    def init_profiles(self, size):
        self.profiles = [[] for i in range(size)]
            
    def update_profiles(self, idxs, batch_size):
        # print('Updating profiles...')
        for idx in idxs:
            ex_items = ', '.join(self.watched_items[idx%batch_size][-self.his_size:])
            if len(self.profiles[idx]) == 0:
                query = [[{"role": "user", "content": f'The user profiles include gender, age, traits, career, interests, and behavioral features. The traits describe the user\'s personality, such as being "compassionate", "ambitious", or "optimistic". The interests indicate the user\'s preferences on the items, such as "sci-fi movies" or "comedy videos". You are a user with your own hobbies, and you have saw {ex_items}. Now generate your profile randomly according to the history and only return the profile content with json format.'}]]
            else:
                json_str = json.dumps(self.profiles[idx])
                query = [[{"role": "user", "content": f'The user profiles include gender, age, traits, career, interests, and behavioral features. The traits describe the user\'s personality, such as being "compassionate", "ambitious", or "optimistic". The interests indicate the user\'s preferences on the items, such as "sci-fi movies" or "comedy videos". You are a user with your own hobbies, this is some information about you: {json_str}. And you have saw {ex_items}. Now update your profile and there must be something changed. Only return the profile content with json format and don\'t record watched movies.'}]]
            received_data = self.send_rev(query)
            for data in received_data:
                jsons = self.extract_json_from_text(data['generation']['content'])
                if len(jsons) == 1:
                    # print(self.profiles[idx])
                    # print(jsons[0])
                    self.profiles[idx] = jsons[0]
                else:
                    print("Update user profiles error!")
                    print(jsons)
        
    def __call__(self, sorted_items):
        try:
            sorted_items = sorted_items.cpu().numpy()
        except:
            sorted_items = sorted_items.numpy()
        
        query = self.encode(sorted_items)
        # print(query[0])
        received_data = self.send_rev(query)
        selected_indices = self.decode(received_data)
        # shape: [B], index start with 1.
        # print(selected_indices[0])
        return selected_indices
    
    def template(self, titles, genres):
        # output: list shaped same as Batch
        
        def join(ts, gs):
            t = ''
            for i in range(len(ts)):
                t += f'{i+1}. {ts[i]}({gs[i]}); '
            return t
        
        texts = np.vectorize(join, signature='(n),(n)->()')(titles, genres)
        
        if self.mem is None:
            output = []
            for idx, text in enumerate(texts):
                # sys = {"role": "system", "content": ""}
                if len(self.profiles[idx]) == 0:
                    json_str = 'None'
                else:
                    json_str = json.dumps(self.profiles[idx])
                i = random.randint(0, 4)
                output.append([{"role": "system", "content": f"This is some information about you: {json_str}. "},
                            {"role": "user", "content": f'This is the viewing history of a certain user: {self.history[i]}. You are a user with your own hobbies. Now, I want to recommend movies to you. Select one that you are most interested in. Only answer the title! {text}'}])
        else:
            output = []
            for idx, text in enumerate(texts):
                # print(self.mem[i])
                if len(self.profiles[idx]) == 0:
                    json_str = 'None'
                else:
                    json_str = json.dumps(self.profiles[idx])
                ex_items = ', '.join(self.watched_items[idx][-self.his_size:])
                i = random.randint(0, 4)
                output.append([{"role": "system", "content": f"This is some information about you: {json_str}. "},
                               {"role": "user", "content": f"This is the viewing history of a certain user: {self.history[i]}. You are a user with your own hobbies, and you have saw {ex_items}. Now, I want to recommend a few more movies to you. Choose a movie that related to your watching history and profile, and have not watched yet. Only answer with the title, no other words. {text}"}])
                
        if self.mem is None:
            self.mem = [[] for i in range(len(titles))]
        if self.watched_items is None:
            self.watched_items = [[] for i in range(len(titles))]
        return output