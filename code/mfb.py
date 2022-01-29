import torch
import torch.nn as nn
import torch.nn.functional as F


class mfb(nn.Module):
    def __init__(self, args, mfb_factor_num, mfb_out_dim, x_dim, y_dim):
        super(mfb, self).__init__()
        self.args = args
        self.MFB_FACTOR_NUM = mfb_factor_num  # k
        self.MFB_OUT_DIM = mfb_out_dim
        self.JOINT_EMB_SIZE = self.MFB_FACTOR_NUM * self.MFB_OUT_DIM
        self.NUM_QUESTION_GLIMPSE = 1
        self.Linear_dataproj = nn.Linear(x_dim, self.JOINT_EMB_SIZE)
        self.Linear_yproj = nn.Linear(y_dim, self.JOINT_EMB_SIZE)
        

    def forward(self, x, y):
       
        data_out = F.dropout(x, self.args.dropout_rate, training=self.training)
        data_out = self.Linear_dataproj(data_out)               
        y = self.Linear_yproj(y)     
        iq = torch.mul(data_out, y)
        iq = F.dropout(iq, self.args.MFB_DROPOUT_RATIO, training=self.training)
        iq = iq.view(-1, 1, self.MFB_OUT_DIM, self.MFB_FACTOR_NUM)
        iq = torch.squeeze(torch.sum(iq, 3))                        # sum pool
        iq = iq.view(self.args.numNWay, self.args.numKShot)
        iq = torch.softmax(iq, dim=1)
        iq = iq.view(-1, 1)
      
        res = iq * x
        
        return res
       
