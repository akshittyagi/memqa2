import pickle as pkl
import os
import json
import random
import torch
from torch.autograd import Variable
import torch.nn.functional as F 
import numpy as np
from utils import to_cuda
def formData(ques, answerChoices, answerChoice, word_to_index):
    
    ques_idxs = [word_to_index[w] for w in ques.split()]
    choices = []
    for choice in answerChoices:
        if choice.strip() == "":
            import ipdb; ipdb.set_trace()
        ans_idxs = [word_to_index[w] for w in choice.split()]
        choices.append(Variable(to_cuda(torch.LongTensor(ans_idxs))))
    
    answerChoice = ord(answerChoice)-ord('A')
    # for CEL-> class label/answerchoice 
    # label = [0]*4
    # label[answerChoice] = 1
    label = answerChoice
    return Variable(to_cuda(torch.LongTensor(ques_idxs))), choices, Variable(to_cuda(torch.LongTensor([label])))

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
            padLength = maxLength - len(answerChoicesB[i][j])
            # print "Max, pad", maxLength, padLength
            currChoices.append(torch.nn.ConstantPad1d((0,padLength),pad_index) (answerChoicesB[i][j]))

        padChoicesB.append(torch.stack(currChoices, dim=0))
    return torch.stack(padChoicesB, dim = 0)

def train(model, train_data, n_epoch, word_to_index, memory, loss_function, optimizer, batch_size=16):
    print "In Training"

    train_data_four = []
    train_data_three = []
    for element in train_data:
        curr = element.split('+')[1].split('/')
        assert(len(curr)==3 or len(curr)==4)
        if len(curr) == 3:
            train_data_three.append(element)
        elif len(curr) == 4:
            train_data_four.append(element)

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
                for data in trainBatch:
                    currData = data.split('+')
                    ques = currData[0]
                    answerChoices = currData[1].split('/')
                    answerChoice = currData[2]
                    ques, answerChoices, answerChoice = formData(ques, answerChoices, answerChoice, word_to_index)
                    quesB.append(ques)
                    answerChoicesB.append(answerChoices)
                    answerChoiceB.append(answerChoice)
                    
                padAnswerChoices = pad_answers(answerChoicesB, word_to_index['<pad>'])
                padQuestions = pad_questions(quesB, word_to_index['<pad>'])
                answerChoiceB = torch.stack(answerChoiceB, dim=0)
                pred_batch = model(memory, padQuestions, padAnswerChoices)
                loss = loss_function(pred_batch, answerChoiceB.squeeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print "Loss", loss

def test(model, data, w2i, batch_size, task_id):
    model.eval()
    correct = 0
    count = 0
    data = []
    for i in range(0, len(data)-batch_size, batch_size):
        batch_data = data[i:i+batch_size]
        story = [d[0] for d in batch_data]
        q = [d[1] for d in batch_data]
        a = [d[2][0] for d in batch_data]

        story_len = min(max_story_len, max([len(s) for s in story]))
        s_sent_len = max([len(sent) for s in story for sent in s])
        q_sent_len = max([len(sent) for sent in q])

        vec_data = vectorize(batch_data, w2i, story_len, s_sent_len, q_sent_len)
        story = [d[0] for d in vec_data]
        q = [d[1] for d in vec_data]
        a = [d[2][0] for d in vec_data]

        story = to_var(torch.LongTensor(story))
        q = to_var(torch.LongTensor(q))
        a = to_var(torch.LongTensor(a))
        pred = model(story, q)
        pred_idx = pred.max(1)[1]
        correct += torch.sum(pred_idx == a).data[0]
        data.append(pred)
    
    pkl.dump(open(path+"_FORMATTED.pkl", 'w'), data)
