import torch
import torch.nn as nn
import math 

class Selfattention(nn.Module):
    def __init__(self,dropout=0.1):
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.softmax=nn.Softmax(dim=-1)

    def forward(self,Q,K,V,mask=None):
        d_k=Q.size(-1)
        scores=torch.matmul(Q,K.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask==0,float('-inf'))
        atten=self.softmax(scores)
        atten=self.dropout(atten)
        out=torch.matmul(atten,V)
        return out,atten

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

        self.attention = Selfattention(dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        Q = self.W_q(q).view(
            batch_size, -1, self.n_heads, self.d_k
        ).transpose(1, 2)

        K = self.W_k(k).view(
            batch_size, -1, self.n_heads, self.d_k
        ).transpose(1, 2)

        V = self.W_v(v).view(
            batch_size, -1, self.n_heads, self.d_k
        ).transpose(1, 2)

        out, atten = self.attention(Q, K, V, mask)

        out = out.transpose(1, 2).contiguous().view(
            batch_size, -1, self.n_heads * self.d_k
        )

        out = self.fc(out)
        out = self.dropout(out)

        return self.norm(out + q), atten



    


