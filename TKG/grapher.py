import json
import re

import numpy as np


class Grapher(object):
    def __init__(self, dataset_dir):
        """
        Store information about the graph (train/valid/test set).
        Add corresponding inverse quadruples to the data.

        Parameters:
            dataset_dir (str): path to the graph dataset directory

        Returns:
            None
        """
        self.dataset_dir = dataset_dir
        self.entity2id = json.load(open(dataset_dir + "entity2id.json"))
        # entity_dict = json.load(open(dataset_dir + "entity2id.json"))
        # for k,v in entity_dict.items():
        #     k = k.replace('"','')
        #     k = k.replace("'",'')
        #     self.entity2id[k]=v
        self.relation2id_old = json.load(open(dataset_dir + "relation2id.json"))
        self.relation2id = self.relation2id_old.copy()
        counter = len(self.relation2id_old)
        for relation in self.relation2id_old:
            self.relation2id["_" + relation] = counter  # Inverse relation
            counter += 1
        self.ts2id = json.load(open(dataset_dir + "ts2id.json"))
        self.id2entity = self.format_dict(self.entity2id)
        self.id2relation = self.format_dict(self.relation2id)
        self.id2ts = self.format_dict(self.ts2id)

        self.inv_relation_id = dict()
        self.num_rels = len(self.relation2id_old)
        for i in range(self.num_rels):
            self.inv_relation_id[i] = i + self.num_rels
        for i in range(self.num_rels, self.num_rels * 2):
            self.inv_relation_id[i] = i % self.num_rels

        self.train_idx = self.create_store("train.txt")
        self.valid_idx = self.create_store("valid.txt")
        self.test_idx = self.create_store("test.txt")
        self.all_idx = np.vstack((self.train_idx, self.valid_idx, self.test_idx))

        print("Grapher initialized.")

    def format_dict(self, origin_dict):
        target_dict = {}
        for k,v in origin_dict.items():
            # if k[0]==['_']:
            #     key = k.split('_')
            #     key = ' '.join(key[1:])
            #     target_dict[v] = key.reverse()
            # else:
            key = k.split('_')
            key = ' '.join(key)
            target_dict[v] = key
        return target_dict

    def create_store(self, file):
        """
        Store the quadruples from the file as indices.
        The quadruples in the file should be in the format "subject\trelation\tobject\ttimestamp\n".

        Parameters:
            file (str): file name

        Returns:
            store_idx (np.ndarray): indices of quadruples
        """

        with open(self.dataset_dir + file, "r", encoding="utf-8") as f:
            quads = f.readlines()
        store = self.split_quads(quads)
        store_idx = self.map_to_idx(store)
        store_idx = self.add_inverses(store_idx)

        return store_idx

    def split_quads(self, quads):
        """
        Split quadruples into a list of strings.

        Parameters:
            quads (list): list of quadruples
                          Each quadruple has the form "subject\trelation\tobject\ttimestamp\n".

        Returns:
            split_q (list): list of quadruples
                            Each quadruple has the form [subject, relation, object, timestamp].
        """

        split_q = []
        for quad in quads:
            split_q.append(quad[:-1].split("\t"))

        return split_q

    def map_to_idx(self, quads):
        """
        Map quadruples to their indices.

        Parameters:
            quads (list): list of quadruples
                          Each quadruple has the form [subject, relation, object, timestamp].

        Returns:
            quads (np.ndarray): indices of quadruples
        """

        subs = [self.entity2id[x[0]] for x in quads]
        rels = [self.relation2id[x[1]] for x in quads]
        objs = [self.entity2id[x[2]] for x in quads]
        tss = [self.ts2id[x[3]] for x in quads]
        quads = np.column_stack((subs, rels, objs, tss))

        return quads

    def add_inverses(self, quads_idx):
        """
        Add the inverses of the quadruples as indices.

        Parameters:
            quads_idx (np.ndarray): indices of quadruples

        Returns:
            quads_idx (np.ndarray): indices of quadruples along with the indices of their inverses
        """

        subs = quads_idx[:, 2]
        rels = [self.inv_relation_id[x] for x in quads_idx[:, 1]]
        objs = quads_idx[:, 0]
        tss = quads_idx[:, 3]
        inv_quads_idx = np.column_stack((subs, rels, objs, tss))
        quads_idx = np.vstack((quads_idx, inv_quads_idx))

        return quads_idx

class IndexGraph(object):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.entity2id,self.relation2id_old = self.load_text2id(dataset_dir)
        self.relation2id = self.relation2id_old.copy()
        counter = len(self.relation2id_old)
        for relation in self.relation2id_old:
            self.relation2id["_" + relation] = counter  # Inverse relation
            counter += 1
        self.id2entity = self.format_dict(self.entity2id)
        self.id2relation = self.format_dict(self.relation2id)

        self.num_ents = len(self.entity2id)
        self.num_rels = len(self.relation2id_old)
        self.inv_relation_id = dict()
        for i in range(self.num_rels):
            self.inv_relation_id[i] = i + self.num_rels
        for i in range(self.num_rels, self.num_rels * 2):
            self.inv_relation_id[i] = i % self.num_rels

        self.train_idx = self.load_quads('train.txt')
        self.valid_idx = self.load_quads('valid.txt')
        self.test_idx = self.load_quads('test.txt')
        self.id2ts = {}

        self.all_idx = np.vstack((self.train_idx, self.valid_idx, self.test_idx))

        print("IndexGraph initialized.")

    def extract_text(self,s):
        # 使用正则表达式查找第一个"<"和第一个">"之间的内容
        match = re.search(r'<(.*?)>', s)

        # 如果找到匹配的内容，返回第一个分组
        if match:
            return match.group(1)

        # 如果没有找到匹配内容，返回空字符串
        return ''

    def format_dict(self, origin_dict):
        target_dict = {}
        for k,v in origin_dict.items():
            if k[0]=='<':
                key = self.extract_text(k)
                key = key.split('_')
                key = ' '.join(key)
                target_dict[v] = key
            else:
                key = k.split('_')
                key = ' '.join(key)
                target_dict[v] = key
        return target_dict

    def load_text2id(self, dataset_dir):
        with open(dataset_dir + 'entity2id.txt', 'r', encoding='utf-8') as f:
            entity_text = f.readlines()
        with open(dataset_dir + 'relation2id.txt', 'r', encoding='utf-8') as f:
            relation_text = f.readlines()
        entity2id = {}
        relation2id = {}
        for line in entity_text:
            entity, id = line.strip().split('\t')[:2]
            entity2id[entity] = int(id)
        for line in relation_text:
            relation, id = line.strip().split('\t')[:2]
            relation2id[relation] = int(id)

        entity2id = dict(sorted(entity2id.items(), key=lambda item: item[1]))
        relation2id = dict(sorted(relation2id.items(), key=lambda item: item[1]))


        return entity2id, relation2id

    def load_quads(self,file):
        with open(self.dataset_dir+file,'r',encoding='utf-8') as f:
            data = f.readlines()
        quads = []
        for line in data:
            s,r,o,t = list(map(int,line.strip().split('\t')[:4]))
            quads.append([s,r,o,t])
            quads.append([o,self.inv_relation_id[r],s,t])

        return np.array(quads)
