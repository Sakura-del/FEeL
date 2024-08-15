import re

from openai import OpenAI
import json
from tqdm import tqdm

from TKG.grapher import IndexGraph,Grapher

dataset = 'icews14'
data_dir = "../data/" + dataset + "/"
# with open(f'{data_dir}/relation2id.json','r') as f:
#     relation2id = json.load(f)
if dataset in ['WIKI','YAGO']:
    data = IndexGraph(data_dir)
else:
    data = Grapher(data_dir)

relation2id = data.relation2id_old
id2relation = data.format_dict(relation2id)

# id2relation = dict()
# for k,v in relation2id.items():
#     t = k.split('_')
#     key = ' '.join(t)
#     id2relation[v] = key


# 注意服务端端口，因为是本地，所以不需要api_key
ip = '127.0.0.1'
#ip = '192.168.1.37'
client = OpenAI(base_url="http://{}:11434/v1".format(ip),
         api_key="ollama")

# 对话历史：设定系统角色是一个只能助理，同时提交“自我介绍”问题
history = [
    {"role": "system", "content":
        "You will act as an expert linguist to help me understand the nature of various social and political relations. I will provide you with a relaiton consisting of one or several words. You will generate a purely descriptive explanation of the relation and an intuitive description of its reverse relation briefly with one sentence each. You must output strictly according to the following format:\n[Relation]<SEP>[Relation's Description]<SEP>[Description of Reverse Relation].\n For example, Accuse<SEP>To claim that someone is responsible for a wrongdoing or illegal act.<SEP>To be charged or blamed by someone for a wrongdoing or illegal act.||im_end|>"}]
# history = [
#     {"role": "system", "content": "You will act as an expert linguist to help me understand the nature of various social and political relationships represented by given phrases. I will provide you with phrases consisting of one or several words that represent relationships. For each phrase, you will generate a detailed and purely descriptive explanation of the relationship and its reverse relationship in one sentence. The output format should be: [Relation]: [Relation description]; [Reverse relation description]. Please ensure that your explanations are clear, concise, and capture the essence of both the relationship and its reverse. "}]

# client = OpenAI(api_key="sk-jWVmcQdth6pKL05A4e7c02913a9845008bDc0dAeC14b00E1",base_url='https://gf.gpt.ge/v1/') # 如果想在代码中设置Api-key而不是全局变量就用这个代码

relation2text =  dict()
num_rel = len(relation2id)

def truncate_string(s):
    # 定义需要截断的字符集合
    truncate_chars = r'[<|\{\[\r\n]'
    # 使用正则表达式查找第一个出现的截断字符的位置
    match = re.search(truncate_chars, s)
    if match:
        # 截断字符串，保留第一个匹配字符之前的部分
        return s[:match.start()]
    else:
        # 如果没有匹配字符，则返回原字符串
        return s

epoch = 0

while len(relation2text) <2*num_rel and epoch <10:
    temp_dict = dict()
    for key,value in tqdm(id2relation.items()):
        history.append({"role": "user", "content": f"{value}"})
        completion = client.chat.completions.create(
            model="llama3.1",
            messages=history,
            temperature=0.5+epoch*0.05,
            stream=True,
            max_tokens=100
        )
        new_message = {"role": "system", "content": ""}

        for chunk in completion:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
                new_message["content"] += chunk.choices[0].delta.content

        history.pop()
        text_list = new_message['content'].split('<SEP>')
        if len(text_list)==3:
            relation2text[int(key)] = text_list[1]
            reverse_desc = text_list[2]
            if ':' in reverse_desc:
                relation2text[int(key) + num_rel] = ' '.join(text_list[2].split(':')[1:])
            elif len(reverse_desc) < 20:
                temp_dict[key] = value
            elif '<' in reverse_desc or '\r' in reverse_desc or '\n' in reverse_desc:
                temp_dict[key] = value
            else:
                relation2text[int(key)+num_rel] = truncate_string(text_list[2])
        else:
            temp_dict[key] = value
    id2relation = temp_dict
    epoch +=1

for k,v in id2relation.items():
    relation2text[int(k)] = v
    relation2text[int(k)+num_rel] = ' '.join(v.split(' ')[::-1])

# with open(f'{data_dir}/unformat_relation2text.json','w') as f:
#     json.dump(id2relation,f)

with open(f'{data_dir}/relation2text_llama3.1.json','w') as f:
    json.dump(relation2text,f)