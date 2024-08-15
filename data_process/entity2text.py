import os
import re
import time
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from openai import OpenAI

from get_wiki_scheme import query_wiki
from TKG.grapher import  Grapher,IndexGraph
import json
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

dataset = 'icews14'
data_dir = "../data/" + dataset + "/"
if dataset in ['WIKI','YAGO']:
    data = IndexGraph(data_dir)
else:
    data = Grapher(data_dir)

id2entity= data.id2entity

# # 注意服务端端口，因为是本地，所以不需要api_key
ip = '127.0.0.1'
#ip = '192.168.1.37'
client = OpenAI(base_url="http://{}:11434/v1".format(ip),
         api_key="ollama")

# client = OpenAI(api_key="sk-jWVmcQdth6pKL05A4e7c02913a9845008bDc0dAeC14b00E1",base_url='https://gf.gpt.ge/v1/') # 如果想在代码中设置Api-key而不是全局变量就用这个代码
# client = OpenAI()

# completion = client.chat.completions.create(
#   model="gpt-3.5-turbo",
#   messages=[
#     {"role": "system", "content": "你是一个机器人"},
#     {"role": "user", "content": "你好"}
#   ]
# )


# 对话历史：设定系统角色是一个只能助理，同时提交“自我介绍”问题
history = [
    {"role": "system", "content": "Your task is to classify a noun based on social and political attributes. The input may be names of people, places, or organizations. Classify the noun into appropriate categories based on its attributes, such as politicians, doctors, government, people, etc. I’ll give you the noun you should classify, please give the most reasonable classfication of it and description of it with less than ten words. You must output strictly according to the following example format: Donald Trump;Politician;The 45th President of the United States. If you don’t know how to classify, please answer in the following format: Al-Shabaab;Unknown;Unknown."}
]

entity2text =  dict()

epoch = 0

num_entity = len(id2entity)

def truncate_string(s):
    # 定义需要截断的字符集合
    truncate_chars = r'[.<|\{\[\r\n]'
    # 使用正则表达式查找第一个出现的截断字符的位置
    match = re.search(truncate_chars, s)
    if match:
        # 截断字符串，保留第一个匹配字符之前的部分
        return s[:match.start()]
    else:
        # 如果没有匹配字符，则返回原字符串
        return s

def is_single_word(s):
    # 去除字符串两端的空格
    s = s.strip()
    # 检查字符串中是否包含空格
    if ' ' in s:
        return False
    else:
        return True

while len(entity2text) < num_entity and epoch <5:
    temp_dict = dict()
    for key, value in tqdm(id2entity.items()):
        history.append({"role": "user", "content": f"{value}"})
        completion = client.chat.completions.create(
            model="llama3.1",
            messages=history,
            temperature=0.5+0.1*epoch,
            stream=True,
            max_tokens=50
        )

        new_message = {"role": "system", "content": ""}

        for chunk in completion:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
                new_message["content"] += chunk.choices[0].delta.content

        # history.append(new_message)
        history.pop()
        text_list = new_message['content'].split(';')
        if len(text_list) != 3:
            temp_dict[int(key)] = value
            continue
        elif text_list[1]=='Unknown' or is_single_word(truncate_string(text_list[2])):
            temp_dict[int(key)] = value
            continue
        entity2text[int(key)] = {'type':text_list[1],'description':truncate_string(text_list[2])}
    id2entity = temp_dict
    epoch +=1

with open(f'{data_dir}/entity2text.json','w') as f:
    json.dump(entity2text,f)

with open(f'{data_dir}/unformat_entity2text.json','w') as f:
    json.dump(id2entity,f)

# with open(f'{data_dir}/entity2text.json','r') as f:
#     entity2text= json.load(f)
#
# with open(f'{data_dir}/unformat_entity2text.json','r') as f:
#     id2entity = json.load(f)

for key, value in tqdm(id2entity.items()):
    history.append({"role": "user", "content": f"{value}"})
    completion = client.chat.completions.create(
        model="llama3.1",
        messages=history,
        temperature=0.6,
        stream=True,
        max_tokens=50
    )
    new_message = {"role": "system", "content": ""}

    for chunk in completion:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
            new_message["content"] += chunk.choices[0].delta.content

    # history.append(new_message)
    history.pop()
    text_list = new_message['content'].split(';')
    if len(text_list) != 3:
        time.sleep(0.5)
        scheme,desc = query_wiki(value)
        if scheme:
            entity2text[int(key)] = {'type':scheme,'description':desc}
        else:
            entity2text[int(key)] = {'type': 'Unknown', 'description': 'Unknown'}
    else:
        if text_list[1]=='Unknown' or is_single_word(truncate_string(text_list[2])):
            time.sleep(0.5)
            scheme, desc = query_wiki(value)
            if scheme:
                entity2text[int(key)] = {'type': scheme, 'description': desc}
            else:
                entity2text[int(key)] = {'type': 'Unknown', 'description': 'Unknown'}
        else:
            entity2text[int(key)] = {'type':text_list[1],'description':truncate_string(text_list[2])}

with open(f'{data_dir}/entity2text.json','w') as f:
    json.dump(entity2text,f)