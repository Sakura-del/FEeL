import torch.nn as nn
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from peft import LoraConfig, TaskType
from peft import get_peft_model
import numpy as np
from ta_gru import AttentionGRU

class SentenceTransformer(nn.Module):
    def __init__(self,tokenizer_name='sentence-transformers/all-mpnet-base-v2',model_name='sentence-transformers/all-mpnet-base-v2',device='cpu'):
        super(SentenceTransformer,self).__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        # peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32,target_modules=['q','k','v'],lora_dropout=0.1)
        # self.model = get_peft_model(self.model, peft_config)
    def forward(self,sentences):
        encoded_input = self.tokenize(sentences).to(self.device)
        model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

    def mean_pooling(self,model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def tokenize(self,sentences):
        tokens=self.tokenizer(list(sentences), padding=True, truncation=True, return_tensors='pt')
        return tokens

class BERTEncoder(nn.Module):
    def __init__(self,tokenizer_name='bert-base-uncased',model_name='bert-base-uncased',device='cpu'):
        super(BERTEncoder, self).__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)

        # peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32,target_modules=['q','k','v'],lora_dropout=0.1)
        # self.model = get_peft_model(self.model_init, peft_config)

    def encode_entity(self,sentences):
        encoded_input = self.tokenize(sentences).to(self.device)
        input_ids = encoded_input['input_ids']
        tokens = [self.tokenizer.convert_ids_to_tokens(input_ids[i]) for i in range(len(sentences))]
        word_indexs = [tokens[i].index(':') + 1 for i in range(len(sentences))]

        model_output = self.model(**encoded_input)
        # 获取目标单词的嵌入表示
        word_embedding = [model_output[0][i, word_indexs[i], :] for i in range(len(sentences))]
        word_embedding = torch.stack(word_embedding,dim=0)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings,word_embedding

    def encode_relation(self,sentences):
        encoded_input = self.tokenize(sentences).to(self.device)
        model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

    def mean_pooling(self,model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def tokenize(self,sentences):
        tokens=self.tokenizer(list(sentences), padding=True, truncation=True, return_tensors='pt')
        return tokens

    def forward(self):
        pass


class TimeEncode(torch.nn.Module):  # time encoder type = 1
    # INDUCTIVE REPRESENTATION LEARNING ON TEMPORAL GRAPHS
    def __init__(self, expand_dim):
        super(TimeEncode, self).__init__()

        time_dim = expand_dim
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        # print(self.basis_freq.shape) #[50]
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())
        self.gru = nn.GRU(time_dim, time_dim, batch_first=True)
        self.comb = nn.Linear(in_features=time_dim*2,out_features=time_dim,bias=True)
        # print(self.phase.shape)#[50]

    def forward(self,rs, pad_ts,length):
        # length = [len(i) for i in ts]
        # pad_ts = nn.utils.rnn.pad_sequence(ts,batch_first=True,padding_value=-1)
        map_ts = pad_ts.unsqueeze(-1) * self.basis_freq  # [N, L, time_dim]
        map_ts += self.phase
        harmonic = torch.cos(map_ts)
        packed_ts = nn.utils.rnn.pack_padded_sequence(harmonic,length,batch_first=True,enforce_sorted=False)
        _,ts = self.gru(packed_ts)
        ts = self.comb(torch.concatenate((rs,ts.squeeze(0)),dim=-1))
        return ts

class TKG_Model(nn.Module):
    def __init__(self,tokenizer_name,model_name,num_rels,device,h_dim=768):
        super(TKG_Model,self).__init__()
        # self.encoder = SentenceTransformer(tokenizer_name,model_name,device)
        self.encoder = BERTEncoder(tokenizer_name,model_name,device)
        # self.encoder = RecurrentRGCN(num_ents,num_rels,h_dim,device)
        self.h_dim = h_dim
        self.device = device
        self.time_encoder = TimeEncode(self.h_dim).to(device)
        weight = torch.ones(num_rels,requires_grad=True) * 0.5
        self.weight = nn.Parameter(weight).to(self.device)

    def forward(self,sentences,timestamps):
        sentence_embdings = self.encoder(sentences)
        # timestamps = [torch.tensor(t).cuda(device=self.device) for t in timestamps]
        timestamps = [torch.tensor(t).cuda(device=self.device) for t in timestamps]
        sentence_embdings = self.time_encoder(sentence_embdings.squeeze(),timestamps)
        # sentence_embdings = self.time_encoder(sentence_embdings,
        #                                       torch.tensor(timestamps, dtype=torch.long).squeeze().cuda(
        #                                           device=self.device))
        return sentence_embdings

class TKGModel(nn.Module):
    def __init__(self,tokenizer_name,model_name,num_ents,num_rels,device,h_dim=768):
        super(TKGModel,self).__init__()
        self.textencoder = BERTEncoder(tokenizer_name,model_name,device)
        self.h_dim = h_dim
        self.device = device
        self.time_encoder = TimeEncode(self.h_dim).to(device)
        weight = torch.ones(num_rels,requires_grad=True) * 0.5
        self.weight = nn.Parameter(weight).to(self.device)
        self.num_ents = num_ents
        self.num_rels = num_rels
        self.comb = nn.Linear(in_features=h_dim*2,out_features=h_dim,bias=True).to(self.device)
        self.fc = nn.Linear(in_features=h_dim*2,out_features=h_dim,bias=True).to(self.device)
        self.emb_entity = torch.nn.Parameter(torch.zeros(self.num_ents +1, self.h_dim), requires_grad=True)
        self.emb_relation = torch.nn.Parameter(torch.zeros(self.num_rels * 2 +1, self.h_dim), requires_grad=True)
        # self.emb_relation = torch.nn.Parameter(torch.Tensor(self.num_rels * 2+1, self.h_dim), requires_grad=True).float().to(self.device)
        # torch.nn.init.xavier_normal_(self.emb_relation)
        #
        # self.emb_entity = torch.nn.Parameter(torch.Tensor(num_ents+1, h_dim), requires_grad=True).float().to(self.device)
        # torch.nn.init.normal_(self.emb_entity)

        self.ta_gru = AttentionGRU(h_dim,h_dim,device=self.device).to(self.device)

    def init_parameter(self,entities,relations):
        all_embeddings = []
        with torch.no_grad():
            batch_size = len(entities)//30
            batches = [entities[i:i + batch_size] for i in range(0, len(entities), batch_size)]

            for batch in tqdm(batches):
                text_emb,class_emb = self.textencoder.encode_entity(batch)
                # entity_emb = self.comb(torch.cat((text_emb,class_emb),dim=-1))
                entity_emb = text_emb
                all_embeddings.append(entity_emb)
        all_embeddings = torch.cat(all_embeddings, dim=0)
        # new_embeddings = torch.nn.Parameter(all_embeddings.clone(),requires_grad=True)
        with torch.no_grad():
            self.emb_entity[1:] = all_embeddings
        with torch.no_grad():
            text_emb = self.textencoder.encode_relation(relations)
            # text_emb = torch.nn.Parameter(text_emb,requires_grad=True)
        with torch.no_grad():
            self.emb_relation[1:] = text_emb

    def forward(self,rel_paths,ent_paths,timestamps):
        rel_length = [len(path) for path in rel_paths]
        # ent_length = [len(path) for path in ent_paths]
        # max_len = max(len(sublist) for sublist in rel_paths)

        # ent_path encoding
        ent_paths = [torch.tensor(sublist) for sublist in ent_paths]
        padded_ent_paths = nn.utils.rnn.pad_sequence(ent_paths,batch_first=True,padding_value=0).to(self.device)
        # input_tensor = torch.tensor(padded_ent_paths).to(self.device)
        embedded_ent_paths = self.emb_entity[padded_ent_paths]
        ent_path_encodings = self.mean_pooling(padded_ent_paths,embedded_ent_paths)

        # rel_path encoding
        rel_paths = [torch.tensor(sublist) for sublist in rel_paths]
        padded_rel_paths = nn.utils.rnn.pad_sequence(rel_paths,batch_first=True,padding_value=0)
        # input_tensor = torch.tensor(padded_ent_paths).to(self.device)
        #
        # padded_rel_paths = [sublist + [0] * (max_len - len(sublist)) for sublist in rel_paths]
        # input_tensor = torch.tensor(padded_rel_paths).to(self.device)
        embedded_rel_paths = self.emb_relation[padded_rel_paths]

        ts = [torch.tensor(sublist) for sublist in timestamps]
        # padded_ts = torch.tensor(padded_ts).to(self.device)
        pad_ts = nn.utils.rnn.pad_sequence(ts,batch_first=True,padding_value=-1).to(self.device)
        output, hidden = self.ta_gru(pad_ts, rel_length, embedded_rel_paths)

        # rel_path_encodings = self.time_encoder(output,pad_ts,rel_length)
        # ablation of ta-gru
        rel_path_encodings = self.time_encoder(output,pad_ts,rel_length)
        path_embdings = self.fc(torch.cat((rel_path_encodings,ent_path_encodings), 1))

        return path_embdings

    def mean_pooling(self, padded_paths, embedded_paths):
        mask = padded_paths !=0
        mask_expanded = mask.unsqueeze(-1).expand(embedded_paths.size()).float()
        return torch.sum(embedded_paths * mask_expanded,1)/torch.clamp(mask_expanded.sum(1),min=1e-9)