import torch
from torch import nn
import torch.nn.functional as F


class Lam(nn.Module):
    def __init__(self,d_model,h):
        super(Lam,self).__init__()
        self.label_attention=nn.MultiheadAttention(d_model,h)
    
    def forward(self,labels,b_s,l_number,word_emb,probs,dims):
        
        label_padding=torch.repeat_interleave(labels.unsqueeze(1), l_number, 1).view(b_s * l_number, -1)

        label_padding=label_padding==1

        labels_emb1=word_emb(labels)
        labels_emb2=torch.multiply(labels_emb1,probs.unsqueeze(-1)).unsqueeze(1).float() ## 
        labels_emb1=torch.repeat_interleave(labels_emb1.unsqueeze(1).float(),l_number,1).view(b_s*l_number,-1,dims).permute(1,0,2)  ## keys without probs
        labels_emb2=torch.repeat_interleave(labels_emb2,l_number,1).view(b_s*l_number,-1,dims).permute(1,0,2)      
        labels_att=self.label_attention(labels_emb1,labels_emb2,labels_emb1,key_padding_mask=label_padding)[0].permute(1,0,2).view(b_s,l_number,-1,dims)  ##
        labels_att=F.sigmoid(labels_att)

        return labels_att
