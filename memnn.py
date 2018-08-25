import pickle as pkl
import os
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import ipdb

from utils import getCombination, to_cuda

class Network(nn.Module):
    def __init__(self, mem_emb_size, embedding_size, vocab_size, hops=2, dropout=0.3):
        super(Network, self).__init__()
        self.hops = hops
        self.embedding_size = embedding_size
        self.lstmLayer = nn.LSTM(embedding_size, embedding_size/2, batch_first=True, bidirectional=True)
        
        self.dropout = nn.Dropout(p=dropout)
        # A -> Memory Representation
        self.A = nn.ModuleList([nn.Embedding(vocab_size, mem_emb_size) for _ in range(hops+1)])
        # B -> Question
        self.B = nn.Embedding(vocab_size, embedding_size)
        # C -> Memory Output Representation
        # self.C = nn.ModuleList([nn.Embedding(vocab_size, embedding_size) for _ in range(hops)])

        #sim for ques, mem
        self.U = Variable(to_cuda(torch.nn.init.xavier_uniform(torch.FloatTensor(3*mem_emb_size,embedding_size)).unsqueeze(0)))
        #sim for answ, mem
        self.V = Variable(to_cuda(torch.nn.init.xavier_uniform(torch.FloatTensor(3*mem_emb_size,embedding_size)).unsqueeze(0)))
        #sim for mem_q, mem_a
        self.W = Variable(to_cuda(torch.nn.init.xavier_uniform(torch.FloatTensor(3*mem_emb_size,3*mem_emb_size)).unsqueeze(0)))

    def embedMemory(self, memTuples, embedder):
        subjectEmb = self.dropout(embedder(memTuples[0])).sum(dim=1)
        relationEmb = self.dropout(embedder(memTuples[1])).sum(dim=1)
        objectEmb = self.dropout(embedder(memTuples[2])).sum(dim=1)

        return [subjectEmb, relationEmb, objectEmb]

    def combineMemory(self, memoryEmb, t='concat'):
        combMemoryEmb = None
        if(t == 'concat'):
            combMemoryEmb = torch.cat(tuple(memoryEmb), dim=1)

        return combMemoryEmb

    def forward(self, memoryBatch, ques, answerChoices, debug=False):
        batch_size = len(ques)
        U = self.U.repeat(batch_size, 1, 1)
        V = self.V.repeat(batch_size, 1, 1)
        W = self.W.repeat(batch_size, 1, 1)

        subjectsBatch = memoryBatch[0]
        relationBatch = memoryBatch[1]
        objectsBatch = memoryBatch[2]

        # assert(len(ques)==len(answerChoices)==len(memoryBatch))
        # allMemIndices = getCombination(memoryBatch, option=6)
        
        # u = self.dropout(self.B(ques)) # u: batch_size * max_qlength * embedding_size
        # _, u = self.lstmLayer(u) 
        # u = torch.cat((u[0][0], u[0][1]), dim=1) # u: batch_size * embedding_size. u[0] - cell state, u[1] - hidden state. u[0][0] - first layer of cell state
        u = self.dropout(self.B(ques)) # u: batch_size * max_qlength * embedding_size
        u = u.sum(dim=1)
        
        answerChoices = answerChoices.transpose(0,1) # Converting from batch_size * 4 * embedding_size --> 4 * batch_size * embedding_size
        a = []
        for choice in answerChoices:
            # a.append(self.C(choice).sum(dim=1))
            a.append(self.B(choice).sum(dim=1))
        o_A = []
        # for hop in range(self.hops):
        #     allMemEmbed = [] 
        #     for mem in allMemIndices[:500]:
        #         allMemEmbed.append(self.dropout(self.A[hop](mem)).sum(dim=0))

        #     allMemEmbed = torch.stack(allMemEmbed, dim=0).unsqueeze(0)
        #     allMemEmbed = allMemEmbed.repeat(batch_size, 1, 1)

        for hop in range(self.hops):
            inpMemBatch= [] 
            outMemBatch= [] 
            for memTuples in zip(*memoryBatch):
                inpMemEmbed = self.combineMemory(self.embedMemory(memTuples, self.A[hop]))
                outMemEmbed = self.combineMemory(self.embedMemory(memTuples, self.A[hop+1]))
                inpMemBatch.append(inpMemEmbed)
                outMemBatch.append(outMemEmbed)
                # inpMemEmbed.append(self.dropout(self.A[hop](memSentences)).sum(dim=1))
                # outMemEmbed.append(self.dropout(self.A[hop+1](memSentences)).sum(dim=1))
            inpMemBatch= torch.stack(inpMemBatch, dim=0)
            outMemBatch= torch.stack(outMemBatch, dim=0)
            p_q = torch.bmm(inpMemBatch,torch.bmm(U, u.unsqueeze(1).transpose(1,2))).squeeze(2) # p_q: batch_size * size_of_memory

            p_q = F.log_softmax(p_q, dim=1) 
            minVals = torch.min(p_q, dim=1)[0].unsqueeze(1)
            p_q = p_q - minVals
            p_q = p_q / (torch.norm(p_q + 1e-8, p=1, dim=1).unsqueeze(1))
        
            #o_q is ques aware mem mebdding
            o_q = torch.bmm(p_q.unsqueeze(1), outMemBatch).squeeze(1) # o_q: batch_size * mem_embedding_size
            u = o_q + u

            a_nextHop = []
            for choice in a:
                p_a = torch.bmm(inpMemBatch, torch.bmm(V, choice.unsqueeze(2))).squeeze(2)
                p_a = F.log_softmax(p_a, dim=1)

                norm = torch.norm(p_a, p=1, dim=1).unsqueeze(1) 
                zero_norm = torch.nonzero(norm == 0)
                if(len(zero_norm) != 0):
                    p_a[zero_norm] = p_a[zero_norm] + 1
                    norm = torch.norm(p_a, p=1, dim=1).unsqueeze(1) 
                p_a = p_a / norm

                #o_a is answ aware mem embedding 
                o_a = torch.bmm(p_a.unsqueeze(1), outMemBatch).squeeze(1) #o_a : batch_sizexmem_emb_size
                a_nextHop.append(o_a + choice) 
                if hop == self.hops - 1:
                    o_A.append(o_a) # o_A : 4*batch_size*mem_emb_size
            a = a_nextHop

        o_A = torch.stack(o_A, dim=0).transpose(0,1).transpose(1,2) # o_A : 4*batch_size*mem_emb_size -> batch_sizex4*mem_embed_size -> batch_size*mem_embed_size*4
        
        prediction = torch.bmm(o_q.unsqueeze(1),torch.bmm(W, o_A)).squeeze(1) # batch_sizex4

        return prediction
        # return F.log_softmax(prediction, dim=1)
        
        



        
