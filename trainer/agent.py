import re
import pickle
import socket
import json
import torch
import random
import numpy as np
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from tqdm import trange


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
        
        self.init_profiles(self.args.num_generated_seqs)
        
        
    def send_rev(self, msg):        
        # 创建TCP socket
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # 连接服务器
        client_socket.connect(('localhost', self.args.port))
        
        # 创建字典
        data_to_send = msg
        
        # print(data_to_send)
        
        # 序列化字典为JSON格式
        data_to_send_json = json.dumps(data_to_send)
        
        # print(data_to_send_json[80:250])
        # 发送数据
        client_socket.send(data_to_send_json.encode())
        # 接收数据
        received_data = client_socket.recv(40960).decode()
        
        # print(received_data[80:250])
        # 关闭连接
        client_socket.close()
        
        received_data = json.loads(received_data)
        
        return received_data
    
    def encode(self, sorted_items):
        # input: converted item id list, start with 1
        # output: list of msg, msg is a list of ditc, a dict likes that: {"role": "user", "content": "XXX"}
        # output = [[
        #     {"role": "user", "content": "XXX"},
        # ]]
        self.org_items = np.vectorize(self.r_smap.get)(sorted_items)
        self.titles = np.vectorize(self.item_info['title'].get)(self.org_items)
        self.genres = np.vectorize(self.item_info['genres'].get)(self.org_items)
        
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
        
        if self.mem is None:
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
                t += f'{i+1}. {ts[i]}({gs[i]}); '
            return t
        
        texts = np.vectorize(join, signature='(n),(n)->()')(titles, genres)
        
        if self.mem is None:
            # sys = {"role": "system", "content": ""}
            output = [[{"role": "user", "content": f'You are a user with your own hobbies. Now, I want to recommend a few more movies to you. Select one that you are most interested in. Only answer the title! {text}'}] for text in texts]
        else:
            output = []
            for i, text in enumerate(texts):
                # print(self.mem[i])
                ex_items = ', '.join(self.watched_items[i])[-50:]
                output.append([{"role": "user", "content": f"You are a user with your own hobbies, and you have saw {ex_items}. Now, I want to recommend a few more movies to you. Choose a movie that you are interested in and have not watched yet. Only answer the title without other words! {text}"}])
                
        if self.mem is None:
            self.mem = [[] for i in range(len(titles))]
        if self.watched_items is None:
            self.watched_items = [[] for i in range(len(titles))]
        return output
    
    def clear(self):
        self.mem = None    
    
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
    
    def init_profiles(self, size):
        pass
    

class ProfileAgent(Agent):
    def init_profiles(self, size):
        def extract_json_from_text(text):
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
        
        self.profiles = []
        print('Initing profiles...')
        profile_query = {"role": "user", "content": f'The user profiles include gender, age, traits, career, interests, and behavioral features. The traits describe the user\'s personality, such as being "compassionate", "ambitious", or "optimistic". The interests indicate the user\'s preferences on the items, such as "sci-fi movies" or "comedy videos". Now generate a random user profile and only return the profile content with json format.'}
        query = None
        for i in trange(size):
            if query is None:
                query = [[profile_query]]
            else:
                query = [query[0][-26:]]
                query[0].append(profile_query)
            received_data = self.send_rev(query)
            query[0].append(received_data[0]['generation'])
            for data in received_data:
                jsons = extract_json_from_text(data['generation']['content'])
                if len(jsons) == 1:
                    self.profiles.append(jsons[0])
                else:
                    raise("Init user profiles error!")
            
    
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
                json_str = json.dumps(self.profiles[idx])
                output.append([{"role": "system", "content": f"This is some information about you: {json_str}. "},
                            {"role": "user", "content": f'You are a user with your own hobbies. Now, I want to recommend a few more movies to you. Select one that you are most interested in. Only answer the title! {text}'}])
        else:
            output = []
            for idx, text in enumerate(texts):
                # print(self.mem[i])
                json_str = json.dumps(self.profiles[idx])
                ex_items = ', '.join(self.watched_items[idx])[-50:]
                output.append([{"role": "system", "content": f"This is some information about you: {json_str}. "},
                               {"role": "user", "content": f"You are a user with your own hobbies, and you have saw {ex_items}. Now, I want to recommend a few more movies to you. Choose a movie that you are interested in and have not watched yet. Only answer the title without other words! {text}"}])
                
        if self.mem is None:
            self.mem = [[] for i in range(len(titles))]
        if self.watched_items is None:
            self.watched_items = [[] for i in range(len(titles))]
        return output