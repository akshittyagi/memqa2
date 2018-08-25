import pickle as pkl
import os
import json
import random

import torch
from torch.autograd import Variable
import torch.nn.functional as F 
import numpy as np

from utils import to_cuda

def toIdxs(text, word_to_index):
    
    idxs = []
    for w in text.split():
        if w in word_to_index:
            idxs.append(word_to_index[w])
        else:
            idxs.append(word_to_index['<unk>'])
    return idxs

def formData(ques, answerChoices, answerChoice, word_to_index):
    
    ques_idxs = toIdxs(ques, word_to_index)
    choices = []
    for choice in answerChoices:
        if choice.strip() == "":
            print "Choice not found for", answerChoices, ques
        ans_idxs = toIdxs(choice, word_to_index)
        choices.append(Variable(to_cuda(torch.LongTensor(ans_idxs))))
    
    answerChoice = ord(answerChoice)-ord('A')
    # for CEL-> class label/answerchoice 
    # label = [0]*4
    # label[answerChoice] = 1
    label = answerChoice
    return Variable(to_cuda(torch.LongTensor(ques_idxs))), choices, Variable(to_cuda(torch.LongTensor([label])))

def formMemory(memoryBatch, word_to_index):
    ret = []
    for elem in memoryBatch:
        idxElem = []
        for sentence in elem:
            idxSentence = []
            for w in sentence.split():
                if w in word_to_index:
                    idxSentence.append(word_to_index[w])
                else:
                    idxSentence.append(word_to_index['<unk>'])
            idxElem.append(to_cuda(torch.LongTensor(idxSentence)))
        ret.append(idxElem)
    
    # import ipdb; ipdb.set_trace()
    return pad_answers(ret, word_to_index['<pad>'])

        
def pad_questions(quesB, pad_index):
    quesLengths = np.array([len(q) for q in quesB])
    maxLength = np.max(quesLengths) + 1
    padLengths = maxLength - quesLengths

    padQuesB = []
    for i in range(len(quesLengths)):
        # print "Max, pad", maxLength, padLengths[i]
        padQuesB.append(torch.nn.ConstantPad1d((0, padLengths[i]),pad_index)(quesB[i]))

    return torch.stack(padQuesB, dim=0)

def pad_answers(answerChoicesB, pad_index):
    flat_list = [item for sublist in answerChoicesB for item in sublist]
    lengths = np.array([len(q) for q in flat_list])
    maxLength = np.max(lengths) + 1

    padChoicesB = []
    for i in range(len(answerChoicesB)):
        currChoices = []
        for j in range(len(answerChoicesB[i])):
            if len(answerChoicesB[i][j]) == 0:
                print "NULL"
                print answerChoicesB[i]
                continue
            padLength = maxLength - len(answerChoicesB[i][j])
            # print "Max, pad", maxLength, padLength
            currChoices.append(torch.nn.ConstantPad1d((0,padLength),pad_index) (answerChoicesB[i][j]))

        padChoicesB.append(torch.stack(currChoices, dim=0))
    print torch.stack(padChoicesB).shape
    return torch.stack(padChoicesB, dim = 0)

def train(model, train_data, n_epoch, word_to_index, memory, loss_function, optimizer, batch_size=16):
    print "In Training"
    training_data = train_data
    relevantMemory = pkl.load(open('relevantMemory.pkl', 'r'))
    pathForRelMem = 'relevantMemory.pkl'
    train_data_four = []
    train_data_three = []
    for key, element in train_data.iteritems():
        curr = element.split('+')[1].split('/')
        assert(len(curr)==3 or len(curr)==4)
        if len(curr) == 3:
            train_data_three.append((key, element))
        elif len(curr) == 4:
            train_data_four.append((key, element))

    for epoch in range(n_epoch):
        print "Epoch", epoch

        for train_data in list([train_data_three, train_data_four]):

            random.shuffle(train_data)
            for i in range(len(train_data)/batch_size):
                print "Batch no", i
                trainBatch = train_data[i*batch_size:(i+1)*batch_size]
                quesB = []
                answerChoicesB = []
                answerChoiceB = []
                relMemoryBatch = []
                for key, data in trainBatch:
                    currData = data.split('+')
                    ques = currData[0]
                    answerChoices = currData[1].split('/')
                    answerChoice = currData[2]
                    ques, answerChoices, answerChoice = formData(ques, answerChoices, answerChoice, word_to_index)
                    quesB.append(ques)
                    answerChoicesB.append(answerChoices)
                    answerChoiceB.append(answerChoice)
                    relMem = relevantMemory[key]
                    relMemoryBatch.append(relMem)
                
                relMemoryBatch = formMemory(relMemoryBatch, word_to_index)

                padAnswerChoices = pad_answers(answerChoicesB, word_to_index['<pad>'])
                padQuestions = pad_questions(quesB, word_to_index['<pad>'])
                answerChoiceB = torch.stack(answerChoiceB, dim=0)
                pred_batch = model(relMemoryBatch, padQuestions, padAnswerChoices)
                loss = loss_function(pred_batch, answerChoiceB.squeeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print "Loss", loss
                # print "PRED BATCH ----------"
                # print pred_batch
                # print "---------------------"
    test(model, training_data, word_to_index, memory, batch_size, pathForRelMem)

def test(model, data, word_to_index, memory, batch_size, pathForRelMem):
    print "In Testing"

    score = 0
    data_four = []
    data_three = []
    dump_data = []

    relevantMemory = pkl.load(open(pathForRelMem, 'r'))

    for key, element in data.iteritems():
        curr = element.split('+')[1].split('/')
        assert(len(curr)==3 or len(curr)==4)
        if len(curr) == 3:
            data_three.append((key, element))
        elif len(curr) == 4:
            data_four.append((key, element))
    
    for data in list([data_three, data_four]):
        random.shuffle(data)
        for i in range(len(data)/batch_size):
            print "Batch no", i
            testBatch = data[i*batch_size:(i+1)*batch_size]
            quesB = []
            answerChoicesB = []
            answerChoiceB = []
            relMemoryBatch = []
            for key, element in testBatch:
                currData = element.split('+')
                ques = currData[0]
                answerChoices = currData[1].split('/')
                answerChoice = currData[2]
                ques, answerChoices, answerChoice = formData(ques, answerChoices, answerChoice, word_to_index)
                quesB.append(ques)
                answerChoicesB.append(answerChoices)
                answerChoiceB.append(answerChoice)
                relMem = relevantMemory[key]
                relMemoryBatch.append(relMem)
            
            relMemoryBatch = formMemory(relMemoryBatch, word_to_index)

            padAnswerChoices = pad_answers(answerChoicesB, word_to_index['<pad>'])
            padQuestions = pad_questions(quesB, word_to_index['<pad>'])
            answerChoiceB = torch.stack(answerChoiceB, dim=0)
            pred_batch = model(relMemoryBatch, padQuestions, padAnswerChoices)
            
            # print pred_batch
            for idx, element in enumerate(pred_batch):
                hits = torch.nonzero(torch.max(element) == element)
                hits = hits.cpu().data.numpy()
                ques = testBatch[idx][1].split('+')
                question_text = ques[0]
                answerChoices = ques[1].split('/')
                answerIndex = ord(ques[2])-ord('A')
                answer = answerChoices[answerIndex]
                if answerIndex in hits:
                    if len(hits) > 1:
                        score += 1.0/len(hits)
                    else:
                        score += 1.0
                dump_data.append(question_text+'+'+ques[1]+'+'+str(element))
                # print answer
                # for val in hits:
                    # for elem in val:
                        # print answerChoices[elem]
    
    print "Score", score
    pkl.dump(dump_data, open("toTemplate.pkl", 'w'))
