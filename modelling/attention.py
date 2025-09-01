import torch
import torch.nn as nn
import math

class Attention(nn.Module):
    def __init__(self, mask_future: bool = False):
        super().__init__()
        self.mask_future = mask_future

    def forward(self, query, key, value, attention_mask):
        
        #inputs:
        # query = (B, SeqQ, D)
        # key = (B, SeqK, D)
        # value = (B, SeqK, D)
        #mask = (B, SeqK)
        # Dimension(D) is same for all
        # Sequence len for key and value are the same(SeqK). for query could be different(SeqQ)
        # Batch size is the same(B)
        # only ket/value get masked so leght of mask is same as key(SeqK)
        # lenght of key = value
        B, SeqQ,D = query.shape
        B, SeqK, D = key.shape
        B, SeqV, D = value.shape

        # S = Q * K_transpose    shape is B * SeqQ * SeqK
        key_t = key.transpose(-2, -1)
        scores = query @ key_t

        # S = S/(D**0.5)
        scores = scores / (D ** 0.5)

        #masking:
            #1 if SeqQ = SeqK, check if 
            # there is future masking and mask top right of scores matrix

            #2 if there is masking for key:
                # for key, when j'th sample is masked, j'th column of scores 
                # would be -inf before applying softmax


        #future:
        if self.mask_future and SeqQ == SeqK:
            for b in range(B):              # loop over batches
                for i in range(SeqQ):       # loop over query positions (rows)
                    for j in range(SeqK):   # loop over key positions (cols)
                        if j > i:           # future position
                            scores[b, i, j] = float("-inf")


        #key masking
        if attention_mask is not None:
            for b in range(B):           # each batch
                for j in range(SeqK):    # each key position
                    if attention_mask[b, j] == 0:
                        scores[b, :, j] = float("-inf")

        # softmax on each row
        weights = torch.zeros_like(scores)
        for b in range(B):            
            for i in range(SeqQ):     # loop over query rows
                row = scores[b, i, :]                
                row_softmax = torch.softmax(row, 0)  
                weights[b, i, :] = row_softmax

        # output = weights * value
        out = weights @ value    
        return out





class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mask_future: bool = False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.mask_future = mask_future
        self.query_transform = nn.Linear(d_model, d_model, bias=False)
        self.key_transform   = nn.Linear(d_model, d_model, bias=False)
        self.value_transform = nn.Linear(d_model, d_model, bias=False)
        self.output_transform= nn.Linear(d_model, d_model, bias=False)
    

    def forward(self, query, key, value, attention_mask):

        B, SeqQ, D = query.shape
        B, SeqK, D = key.shape
        B, SeqV, D = value.shape


        # multplying q,k,v with their transform weights (linear projection)
        Q = self.query_transform(query)   # (B, SeqQ, D)
        K = self.key_transform(key)       # (B, SeqK, D)
        V = self.value_transform(value)   # (B, SeqK, D)

        d_head = D // self.num_heads

        Q_heads, K_heads, V_heads = [], [], []


        # spliting q,k,v into heads
        for h in range(self.num_heads):
            Q_part = Q[:, :, h*d_head:(h+1)*d_head]   # shape (B, SeqQ, d_head)
            K_part = K[:, :, h*d_head:(h+1)*d_head]   # shape (B, SeqK, d_head)
            V_part = V[:, :, h*d_head:(h+1)*d_head]   # shape (B, SeqK, d_head)
            Q_heads.append(Q_part)
            K_heads.append(K_part)
            V_heads.append(V_part)


        # attention for each head:
        head_outputs = []

        for h in range(self.num_heads):
            Q_h = Q_heads[h]  
            K_h = K_heads[h]   
            V_h = V_heads[h] 
            attn = Attention(mask_future=self.mask_future)
            out_h = attn(Q_h, K_h, V_h, attention_mask)  # (B, SeqQ, d_head)
            head_outputs.append(out_h) # after concatenation ends we get (B, SeqQ, d_head)

        

        concatenated = torch.cat(head_outputs, dim=-1)
        out = self.output_transform(concatenated)
        return out






        











