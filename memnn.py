import pickle as pkl
import os
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from utils import getCombination, to_cuda

class Network(nn.Module):
    def __init__(self, mem_emb_size, embedding_size, vocab_size, hops=2, dropout=0.1):
        super(Network, self).__init__()
        self.hops = hops
        self.embedding_size = embedding_size
        self.lstmLayer = nn.LSTM(embedding_size, embedding_size/2, batch_first=True)
        
        self.dropout = nn.Dropout(p=0.1)
        # A -> Memory
        self.A = nn.ModuleList([nn.Embedding(vocab_size, mem_emb_size) for _ in range(hops)])
        # B -> Question
        self.B = nn.Embedding(vocab_size, embedding_size)
        # C -> Answer Choice
        self.C = nn.Embedding(vocab_size, embedding_size)

        torch.nn.init.xavier_uniform(torch.FloatTensor(5,5))
        #sim for ques, mem
        self.U = Variable(to_cuda(torch.nn.init.xavier_uniform(torch.FloatTensor(mem_emb_size,embedding_size)).unsqueeze(0)))
        #sim for answ, mem
        self.V = Variable(to_cuda(torch.nn.init.xavier_uniform(torch.FloatTensor(mem_emb_size,embedding_size)).unsqueeze(0)))
        #sim for mem_q, mem_a
        self.W = Variable(to_cuda(torch.nn.init.xavier_uniform(torch.FloatTensor(mem_emb_size,mem_emb_size)).unsqueeze(0)))

    def forward(self, memory, ques, answerChoices):
        
        batch_size = len(ques)
        U = self.U.repeat(batch_size, 1, 1)
        V = self.V.repeat(batch_size, 1, 1)
        W = self.W.repeat(batch_size, 1, 1)

        assert(len(ques)==len(answerChoices))

        allMemIndices = getCombination(memory, option=6)
        
        
        u = self.dropout(self.B(ques)) # u: batch_size * max_qlength * embedding_size
        _, u = self.lstmLayer(u) 
        u = torch.cat((u[0][0], u[1][0]), dim=1) # u: batch_size * embedding_size. u[0] - cell state, u[1] - hidden state. u[0][0] - first layer of cell state
        
        answerChoices = answerChoices.transpose(0,1) # Converting from batch_size * 4 * embedding_size --> 4 * batch_size * embedding_size
        a = []
        for choice in answerChoices:
            a.append(self.C(choice).sum(dim=1))
        o_A = []
        
        for hop in range(self.hops):
            allMemEmbed = [] 
            for mem in allMemIndices:
                allMemEmbed.append(self.dropout(self.A[hop](mem)).sum(dim=0))

            allMemEmbed = torch.stack(allMemEmbed, dim=0).unsqueeze(0)
            allMemEmbed = allMemEmbed.repeat(batch_size, 1, 1)
            p_q = torch.bmm(allMemEmbed,torch.bmm(U, u.unsqueeze(1).transpose(1,2))).squeeze(2) # p_q: batch_size * size_of_memory
            p_q = F.softmax(p_q, dim=1) 
            #o_q is ques aware mem mebdding
            o_q = torch.bmm(p_q.unsqueeze(1), allMemEmbed).squeeze(1) # o_q: batch_size * mem_embedding_size
            u = o_q + u

            a_nextHop = []
            for choice in a:
                # import ipdb; ipdb.set_trace()
                p_a = torch.bmm(allMemEmbed, torch.bmm(V, choice.unsqueeze(2))).squeeze(2)
                p_a = F.softmax(p_a, dim=1)
                #o_a is answ aware mem embedding 
                o_a = torch.bmm(p_a.unsqueeze(1), allMemEmbed).squeeze(1) #o_a : batch_sizexmem_emb_size
                a_nextHop.append(o_a + choice) 
                if hop == self.hops - 1:
                    o_A.append(o_a) # o_A : 4*batch_size*mem_emb_size
            a = a_nextHop

        o_A = torch.stack(o_A, dim=0).transpose(0,1).transpose(1,2) # o_A : 4*batch_size*mem_emb_size -> batch_sizex4*mem_embed_size -> batch_size*mem_embed_size*4
        
        # import ipdb; ipdb.set_trace()
        prediction = torch.bmm(o_q.unsqueeze(1), torch.bmm(W, o_A)).squeeze(1) # batch_sizex4

        return F.log_softmax(prediction, dim=1)
        
        



        