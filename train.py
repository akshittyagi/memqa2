import pickle as pkl
import os
import json
import random
import torch

def formData(ques, answerChoices, answerChoice, word_to_index):
    
    ques_idxs = [word_to_index[w] for w in ques.split()]
    choices = []
    for choice in answerChoice:
        ans_idxs = [word_to_index[w] for w in choice.split()]
        choices.append(torch.LongTensor(ans_idxs))
    
    answerChoice = ord(answerChoice)-ord('A')
    label = [0]*4
    label[answerChoice] = 1
    return torch.LongTensor(ques_idxs), choices, torch.LongTensor(label)

def train(model, train_data, n_epoch, word_to_index):
    for epoch in range(n_epoch):
        
        random.shuffle(train_data)
        for data in train_data:
            currData = data.split('+')
            ques = currData[0]
            answerChoices = currData[1].split('/')
            answerChoice = currData[2]
            ques, answerChoices, answerChoice = formData(ques, answerChoices, answerChoice, word_to_index)
    

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
