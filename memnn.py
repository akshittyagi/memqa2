import pickle as pkl
import os
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import getCombination

class Network(nn.Module):
    def __init__(self, mem_emb_size, embedding_size, vocab_size, hops=2, dropout=0.1):
        super(Network, self).__init__()
        self.hops = hops
        self.embedding_size = embedding_size
        self.lstmLayer = nn.LSTM(embedding_size, embedding_size/2)
        
        self.dropout = nn.Dropout(p=0.1)
        # A -> Memory
        self.A = nn.ModuleList([nn.Embedding(vocab_size, mem_emb_size) for _ in range(hops)])
        # B -> Question
        self.B = nn.Embedding(vocab_size, embedding_size)
        # C -> Answer Choice
        self.C = nn.Embedding(vocab_size, embedding_size)

        #sim for ques, mem
        self.U = Variable(torch.FloatTensor([embedding_size, mem_emb_size]))
        #sim for answ, mem
        self.V = Variable(torch.FloatTensor([embedding_size, mem_emb_size]))
        #sim for mem_q, mem_a
        self.W = Variable(torch.FloatTensor([mem_emb_size, mem_emb_size]))

    def forward(self, memory, qna):
        
        mem = getCombination(memory, option=6)
        ques = qna[0]
        answerChoices = qna[1]

        print "Going to pdb"
        u = self.dropout(self.B(ques))
        _, u = self.lstmLayer(u.unsqueeze(0))
        import ipdb; ipdb.set_trace()
        u = torch.cat((u[0][-1][-1], u[1][-1][-1]), dim=0).unsqueeze(0)
        a = []
        for choice in answerChoices:
            a.append(self.C(choice).sum(dim=0).unsqueeze(0))
        o_A = []
        import ipdb; ipdb.set_trace()
        
        for hop in range(self.hops):
            mem = self.dropout(self.A[hop](mem))
            p_q = torch.bmm(mem,torch.bmm(self.U, u))
            p_q = F.softmax(p_q, -1) 
            #o_q is ques aware mem mebdding
            o_q = torch.bmm(p_q, mem)
            u = o_q + u

            for choice in a:
                p_a = torch.bmm(mem, torch.bmm(self.V, choice))
                p_a = F.softmax(p_a, -1)
                #o_a is answ aware mem embedding 
                o_a = torch.bmm(p_a, mem)
                choice = o_a + choice
                if hop == self.hops - 1:
                    o_A.append(o_a)

        
        prediction = torch.bmm(o_q, torch.bmm(self.W, o_A))

        return F.log_softmax(prediction, -1)
        
        



        