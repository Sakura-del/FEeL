import numpy as np
import random
from tqdm import tqdm

import os
import argparse
from TKG.grapher import Grapher,IndexGraph
import utils
import pickle

parser = argparse.ArgumentParser(description='Negative Sampling')
parser.add_argument('--dataset', type=str, default='icews14_fs_lg',
                    help='name of the dataset')
parser.add_argument('--output_dir', type=str, default=None,
                    help='directory to store output')
parser.add_argument('--training_mode', type=str, default='train',
                    help='whether train, valid or test')
parser.add_argument('--neg_num', type=int, default=3,
                    help='number of negative samples')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--train_history_len', type=int, default=10,
                    help='length of historical graphs')

args = parser.parse_args()
print(args)
random.seed(args.seed)
dataset = args.dataset
data_dir = "../data/" + dataset + "/"
if dataset in ['WIKI','YAGO']:
    data = IndexGraph(data_dir)
else:
    data = Grapher(data_dir)

if args.training_mode == 'train':
    train_data = data.train_idx
elif args.training_mode == 'valid':
    train_data = data.valid_idx
else:
    train_data = data.test_idx
if args.output_dir is None:
    output_dir = f'../data/path_data/{args.dataset}/'
else:
    output_dir = args.output_dir

# if args.triplet_sample_num!=-1:
#     train_triplets=random.sample(train_triplets,args.triplet_sample_num)
# pbar = tqdm(total=len(train_data), desc=args.training_mode,
#             position=0, leave=True,
#             file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET))
neg_quads = []
data_list = data.all_idx
data_list = np.vstack((data.train_idx,data.valid_idx,data.test_idx))
data_list = utils.split_by_time_with_t(data_list, np.unique(data_list[:, 3]))
train_list = utils.split_by_time_with_t(train_data, np.unique(train_data[:, 3]))
time_list = np.unique(train_data[:,3])
# entities = data.entity2id.keys()
# neighbors = utils.store_history_graph(data.all_idx)
for idx in tqdm(time_list):
    if idx ==0: continue
    if idx - args.train_history_len < 0:
        history_data = data_list[0:idx]
    else:
        history_data = data_list[idx - args.train_history_len:idx]
    history_data = np.vstack(history_data)
    neighbors = utils.store_history_graph(history_data)
    entities = neighbors.keys()

    for t in data_list[idx]:
        neg_quads.append(t.tolist())
        try:
            # neg_tails = set(entities) - set(neighbors[t[0]][:, 2])
            neg_tails = set(entities) - set([t[2]])
        except Exception as e:
            neg_tails = set(entities)
        if len(neg_tails) < args.neg_num:
            print(2)
        neg_tails = random.sample(neg_tails, args.neg_num - 1)
        for n in neg_tails:
            neg_quads.append([t[0], t[1], n, t[3]])
    # if args.finding_mode == 'relation':
    #     for t in data_list[idx]:
    #         neg_quads.append(t.tolist())
    #         try:
    #             neg_rels = set(data.relation2id.values()) - set(neighbors[t[0]][:,1])
    #         except:
    #             neg_rels = set(data.relation2id.values())
    #         neg_rels = random.sample(neg_rels,args.neg_num -1)
    #         for n in neg_rels:
    #             neg_quads.append([t[0],n,t[2],t[3]])

# if args.finding_mode == 'tail':
#     for t in train_data:
#         neg_quads.append(t.tolist())
#         neg_tails = set(entities) - set(neighbors[t[2][:, 0]])
#         neg_tails = random.sample(neg_tails, args.neg_num - 1)
#         for n in neg_tails:
#             neg_quads.append([n, t[1], t[2], t[3]])
#         pbar.update(1)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# with open(os.path.join(output_dir,f'entities.txt'),'w',encoding='utf-8') as f:
#     for e in entities:
#         f.write(e+"\n")
with open(os.path.join(output_dir, f'sample_{args.training_mode}.pickle'), 'wb') as f:
    pickle.dump(np.array(neg_quads), f)

