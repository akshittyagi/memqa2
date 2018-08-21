import pickle as pkl
import os
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

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

        torch.nn.init.xavier_uniform(torch.FloatTensor(5,5))
        #sim for ques, mem
        self.U = Variable(torch.nn.init.xavier_uniform(torch.FloatTensor(mem_emb_size,embedding_size)).unsqueeze(0))
        #sim for answ, mem
        self.V = Variable(torch.nn.init.xavier_uniform(torch.FloatTensor(mem_emb_size,embedding_size)).unsqueeze(0))
        #sim for mem_q, mem_a
        self.W = Variable(torch.nn.init.xavier_uniform(torch.FloatTensor(mem_emb_size,mem_emb_size)).unsqueeze(0))

    def forward(self, memory, qna):
        

        allMemIndices = getCombination(memory, option=6)
        ques = qna[0]
        answerChoices = qna[1]

        print "Going to pdb"
        u = self.dropout(self.B(ques))
        _, u = self.lstmLayer(u.unsqueeze(0))
        u = torch.cat((u[0][-1][-1], u[1][-1][-1]), dim=0).unsqueeze(0)
        a = []
        for choice in answerChoices:
            a.append(self.C(choice).sum(dim=0).unsqueeze(0))
        o_A = []
        
    
        for hop in range(self.hops):
            allMemEmbed = [] 
            for mem in allMemIndices[:50]:
                allMemEmbed.append(self.dropout(self.A[hop](mem)).sum(dim=0))
            allMemEmbed = torch.stack(allMemEmbed, dim=0).unsqueeze(0)
            p_q = torch.bmm(allMemEmbed,torch.bmm(self.U, (u.t()).unsqueeze(0))).squeeze(2)
            p_q = F.softmax(p_q, dim=1) 
            #o_q is ques aware mem mebdding
            o_q = torch.bmm(p_q.unsqueeze(1), allMemEmbed).squeeze(1)
            u = o_q + u

            for choice in a:
                p_a = torch.bmm(allMemEmbed, torch.bmm(self.V, (choice.t()).unsqueeze(0))).squeeze(2)
                p_a = F.softmax(p_a, dim=1)
                #o_a is answ aware mem embedding 
                o_a = torch.bmm(p_a.unsqueeze(1), allMemEmbed).squeeze(1) #o_a : batch_sizexmem_emb_size
                choice = o_a + choice
                if hop == self.hops - 1:
                    o_A.append(o_a) # o_A : 4*batch_size*mem_emb_size

        o_A = torch.stack(o_A, dim=0).transpose(0,1).transpose(1,2) # o_A : 4*batch_size*mem_emb_size -> batch_sizex4*mem_embed_size -> batch_size*mem_embed_size*4 , rem unsqueeze
        
        prediction = torch.bmm(o_q.unsqueeze(0), torch.bmm(self.W, o_A)).squeeze(1) # batch_sizex4

        return F.log_softmax(prediction, dim=1)
        
        



        