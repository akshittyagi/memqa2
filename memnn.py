import pickle as pkl
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import getCombination

class Network(nn.Module):
    def __init__(self, hops=2, dropout=0.1, embedding_size, vocab_size):
        self.hops = hops
        self.embedding_size = embedding_size
        
        self.dropout = nn.Dropout(p=0.1)
        # A -> Memory
        self.A = nn.ModuleList([nn.Embedding(vocab_size, embedding_size) for _ in range(hops)])
        # B -> Question
        self.B = nn.ModuleList([nn.Embedding(vocab_size, embedding_size) for _ in range(hops)])
        # C -> Answer Choice
        self.C = nn.ModuleList([nn.Embedding(vocab_size, embedding_size) for _ in range(hops)])

        #sim for ques, mem
        self.U = nn.Matrix()
        #sim for answ, mem
        self.V = nn.Matrix()
        #sim for mem_q, mem_a
        self.W = nn.Matrix()

    def forward(self, memory, qna):
        
        memory = getCombination(memory, option=1)
        ques = qna[0]
        answerChoices = qna[1]

        u = self.dropout(self.B[0](q))
        a = []
        for choice in answerChoices:
            a.append(self.C[0](choice))

        o_A = []
        for choice in a:
            p_a = torch.bmm(mem, torch.bmm(self.V, choice))
            p_a = F.softmax(p_a, -1)
            #o_a is answ aware mem embedding 
            o_a = torch.bmm(p_a, mem)

            o_A.append(o_a)

        for hops in range(self.hops):
            mem = self.dropout(self.A[hop](memory))
            p_q = torch.bmm(mem,torch.bmm(self.U, u))
            p_q = F.softmax(p_q, -1) 
            #o_q is ques aware mem mebdding
            o_q = torch.bmm(p_q, mem)
            u = o_q + u
        
        prediction = torch.bmm(o_q, torch.bmm(self.W, o_A))

        return F.log_softmax(prediction, -1)
        
        



        