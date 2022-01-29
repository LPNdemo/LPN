import math
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from mfb import mfb

from transformers import BertTokenizer, BertConfig, BertModel



class BertEncoder(nn.Module):
    
    def __init__(self, args):
        super(BertEncoder, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained(args.fileVocab, do_lower_case=False)
        config = BertConfig.from_json_file(args.fileModelConfig)   
        self.bert = BertModel.from_pretrained(args.fileModel,config=config)
        self.device = torch.device('cuda', args.numDevice)
        torch.cuda.set_device(self.device)
        if args.numFreeze > 0:
            self.freeze_layers(args.numFreeze)
  
        self.bert.cuda()

       

    def freeze_layers(self, numFreeze):
        unfreeze_layers = ["pooler"]
        for i in range(numFreeze, 12):
            unfreeze_layers.append("layer."+str(i))

        for name ,param in self.bert.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break
    
    def forward(self, text):

        tokenizer = self.tokenizer(
        text,
        padding = True,
        truncation = True,
        max_length = 250,
        return_tensors='pt'  # 返回的类型为pytorch tensor
        )
        input_ids = tokenizer['input_ids'].to(self.device)
        token_type_ids = tokenizer['token_type_ids'].to(self.device)
        attention_mask = tokenizer['attention_mask'].to(self.device)

        outputs = self.bert(
              input_ids,
              attention_mask=attention_mask,
              token_type_ids=token_type_ids
              )

        last_hidden_states, _, _, _ = outputs
        
        return last_hidden_states

class SelfAttentive(nn.Module):
    def __init__(self, args):
        super(SelfAttentive, self).__init__()
        
        self.linear1 = nn.Linear(768, 256)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(4*768, 768)
        self.relu = nn.ReLU()
        self.context_vector = nn.Parameter(torch.randn(256, 4), requires_grad=True)
    
    def forward(self, last_hidden_states):
        b, _, _ = last_hidden_states.shape
        vectors = self.context_vector.unsqueeze(0).repeat(b, 1, 1)
        h = self.linear1(last_hidden_states) # (b, t, h)
        h = self.tanh(h)
        scores = torch.bmm(h, vectors) # (b, t, 4)
        scores = nn.Softmax(dim=1)(scores) # (b, t, 4)
        outputs = torch.bmm(scores.permute(0, 2, 1), last_hidden_states).view(b, -1) # (b, 4h)
        outputs = self.relu(outputs)
        outputs = self.linear2(outputs)
        return outputs

class QKVAttention(nn.Module):
    """
    Attention mechanism based on Query-Key-Value architecture. And
    especially, when query == key == value, it's self-attention.
    """

    def __init__(self, query_dim, hidden_dim, dropout_rate):
        super(QKVAttention, self).__init__()

        
        self.__query_dim = query_dim
        self.__hidden_dim = hidden_dim
        self.__dropout_rate = dropout_rate

        # Declare network structures.
        self.__query_layer = nn.Linear(self.__query_dim, self.__hidden_dim)
        self.__dropout_layer = nn.Dropout(p=self.__dropout_rate)

    def forward(self, input_query, input_key, input_value):
        
        linear_query = self.__query_layer(input_query)
        linear_key = input_key
        linear_value = input_value
        score_tensor = F.softmax(torch.matmul(
            linear_query,
            linear_key.transpose(-2, -1)
        ) / math.sqrt(self.__hidden_dim), dim=-1)
        forced_tensor = torch.matmul(score_tensor, linear_value)
        forced_tensor = self.__dropout_layer(forced_tensor)

        return forced_tensor



class ContrastiveModel(nn.Module):
    def __init__(self, args):
        super(ContrastiveModel, self).__init__()
        self.args = args
        self.attention = QKVAttention(1536, 768, 0.1)

    def forward(self, query_hidden_states, prototype, label_embedding):
       
        k = torch.cat((prototype, label_embedding), 1)
        query_embeddings = self.attention(k, query_hidden_states, query_hidden_states)
      
        return query_embeddings

class MyModel(nn.Module):
    def __init__(self, args):
        super(MyModel, self).__init__()
        self.args = args
        self.bert = BertEncoder(args)
        self.proto_lpn = mfb(args, args.MFB_FACTOR_NUM, args.MFB_OUT_DIM, 768, 768)
        # Predict the number of labels
        self.linear = nn.Linear(768, args.count)

        self.selfattentive = SelfAttentive(args)
        self.constrastive = ContrastiveModel(args)
        


    def forward(self, text, label_classes):
        support_size = self.args.numNWay * self.args.numKShot
        query_size = self.args.numNWay * self.args.numQShot
        text_hidden_states = self.bert(text)
        label_hidden_states = self.bert(label_classes)  

        text_embedding = self.selfattentive(text_hidden_states)
        label_embeddings = self.selfattentive(label_hidden_states)
       
        support_embddings = text_embedding[:support_size]  # NK X 768
        query_embeddings = text_embedding[support_size:]   # NQ X 768

        count_outputs = self.linear(text_embedding)

        label_em = label_embeddings.repeat(1, self.args.numKShot).view(-1, label_embeddings.shape[1])
        
        prototypes = self.proto_lpn(support_embddings, label_em)  # NK X dim
    
        c_prototypes = prototypes.view(self.args.numNWay, -1, prototypes.shape[1])  # N X K X dim
        c_prototypes = torch.sum(c_prototypes, dim=1)

        instance_constrastive = self.constrastive(text_hidden_states, c_prototypes, label_embeddings)
        
        
        return (c_prototypes, query_embeddings, count_outputs, instance_constrastive)


