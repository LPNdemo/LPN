import torch
from torch.nn import CrossEntropyLoss
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import precision_score,f1_score,recall_score, accuracy_score, roc_auc_score


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def constrastive_loss(features, labels, temperature=0.1, mask=None):
     
    features = F.normalize(features, p=2, dim=1)
    device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

    batch_size = features.shape[0]
    labels = labels.contiguous().view(-1, 1)
    # mask [bsz, bsz], x_ij = 1 if label_i=label_j, and x_ii = 1
    mask = torch.eq(labels, labels.T).float().to(device)


    # compute logits, anchor_dot_contrast: (bsz, bsz), x_i_j: (z_i*z_j)/t
    anchor_dot_contrast = torch.div(
        torch.matmul(features, features.T),
        temperature)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    if torch.any(torch.isnan(log_prob)):
        raise ValueError("Log_prob has nan!")
    

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+1)

    if torch.any(torch.isnan(mean_log_prob_pos)):
        raise ValueError("mean_log_prob_pos has nan!")

    # loss
    loss = - mean_log_prob_pos
    if torch.any(torch.isnan(loss)):
            raise ValueError("loss has nan!")
    loss = loss.mean()

    return loss

def useful_query_embeddings(query_embeddings, labels):
    # query_embeddings: bsz, Nway, dim; labels: bsz, Nway
    # obtain useful query embeddings: bsz, dim; useful labels: bsz
    useful_embeddings, useful_labels = [], []
    bsz = labels.shape[0]
    Nway = labels.shape[1]
    for i in range(bsz):
        query_set = query_embeddings[i]
        label_set = labels[i]
        for j in range(Nway):
            if label_set[j] == 1:
                useful_embeddings.append(query_set[j])
                useful_labels.append(j)

    useful_embeddings = torch.stack(useful_embeddings)
    useful_labels = torch.tensor(useful_labels)
   
    return useful_embeddings, useful_labels

 

class Loss_fn(torch.nn.Module):
    def __init__(self, args):
        super(Loss_fn, self).__init__()
        
        self.loss = CrossEntropyLoss()
        self.args = args

    def forward(self, model_outputs, labels):
        query_size = self.args.numNWay * self.args.numQShot
        support_size = self.args.numNWay * self.args.numKShot

        prototypes, q_re, count_outputs, query_constrative = model_outputs

        # lepn
        dists = euclidean_dist(q_re, prototypes)
        log_p_y = F.log_softmax(-dists, dim=1) # num_query x num_class
        query_labels = labels[support_size:]
        query_labels = torch.tensor(query_labels, dtype=float).cuda()
        labels = torch.tensor(labels).cuda()
        loss1 = - query_labels * log_p_y
        loss1 = loss1.mean()

        # SCL 
        u_query, u_query_labels = useful_query_embeddings(query_constrative, labels)
        scl_loss = constrastive_loss(u_query, u_query_labels, temperature=self.args.temperature)
        # count loss
        labels_count = labels.sum(dim=1)-1
        four = torch.ones_like(labels_count)*4
        labels_count = torch.where(labels_count > 4, four, labels_count)
        count_loss = self.loss(count_outputs, labels_count)

        loss = loss1 + 0.1 * count_loss + 0.01 * scl_loss
       
        
        # multi
        _, count_pred  = torch.max(count_outputs, 1, keepdim=True)
        labels_count = labels_count.cpu().detach()
        count_pred = count_pred.cpu().detach()
        c_acc = accuracy_score(labels_count, count_pred)
        query_count = count_pred[support_size:]

        sorts, indices = torch.sort(log_p_y, descending=True)  
        x = []
        for i, t in enumerate(query_count):
            x.append(log_p_y[i][indices[i][query_count[i][0]]])
        x = torch.tensor(x).view(log_p_y.shape[0], 1).cuda()
        one = torch.ones_like(log_p_y)
        zero = torch.zeros_like(log_p_y)
        y_pred = torch.where(log_p_y >= x, one, log_p_y)
        y_pred = torch.where(y_pred < x, zero, y_pred)

        target_mode = 'macro'

        query_labels = query_labels.cpu().detach()
        y_pred = y_pred.cpu().detach()
        p = precision_score(query_labels, y_pred, average=target_mode)
        r = recall_score(query_labels, y_pred, average=target_mode)
        f = f1_score(query_labels, y_pred, average=target_mode)
        acc = accuracy_score(query_labels, y_pred)
        
        y_score = F.softmax(-dists, dim=1)
        y_score = y_score.cpu().detach()
        auc = roc_auc_score(query_labels, y_score)
        
        
        return loss, p, r, f, acc, auc, c_acc
       
       






