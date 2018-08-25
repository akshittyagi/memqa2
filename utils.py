import os
import torch
import pickle as pkl
import torch.nn as nn
from torch.autograd import Variable
import ipdb
import spacy    

nlp = spacy.load('en')

def tokenize(sentence):
    doc = nlp(sentence.decode('utf8'), disable=['parser', 'tagger', 'ner'])
    return " ".join(token.lemma_ for token in doc)
# def prepareMemory(memory, word_to_index):
    # ipdb.set_trace()
    # ret = dict()
    # for key in memory:
        # tuples = memory[key]
        # tuples = [" ".join(t) for t in tuples]

        # idxs = [word_to_index[w] for w in tuples.split()]
        # ret[key] = Variable(to_cuda(torch.LongTensor(idxs)))
    # return ret

def buildDictionary(sentences):
    word_to_index = dict()
    for sentence in sentences:
        for word in sentence.split():
            if(word not in word_to_index):
                word_to_index[word] = len(word_to_index)

    word_to_index['<pad>'] = len(word_to_index)
    word_to_index['<unk>'] = len(word_to_index)
    return word_to_index

def processSentence(sentence):           
    currSent = sentence.split('+')
    
    question = tokenize(currSent[0])
    answers = [tokenize(a) for a in currSent[1].split('/')]
    gt = currSent[2]

    return question, answers, gt

def getWordsForm(formSentence):
    question, answers, _ = processSentence(formSentence)
    words = list()
    for word in question.split():
        words.append(word)
    for answer in answers:
        for word in answer.split():
            words.append(word)

    return words

def getWordsTuple(tuples):
    words = list()
    for Tuple in tuples:
        for elem in Tuple:
            for word in elem.split():
                words.append(word)
        
    return words

def getCombination(memory, option):
    ret = []
    if option==1:
        idx = 0
        while( idx < len(memory) ):
            temp = memory[idx]
            temp.append(memory[idx+1])
            temp.append(memory[idx+2])
            ret.append(temp)
            idx = idx + 3
    
    elif option==2:
        idx = 0
        while( idx < len(memory) ):
            temp = torch.Variable(init=memory[idx])
            temp += memory[idx+1]
            temp += memory[idx+2]
            ret.append(temp/3)
            idx = idx + 3
    
    elif option==3:
        idx = 0
        while( idx < len(memory) ):
            temp = memory[idx]
            temp2 = memory[idx+1]
            temp3 = memory[idx+2]
            a,b,c = getParams(nn.Module, [temp,temp2,temp3])
            idx = idx + 3
            ret.append(a*temp + b*temp2 + c*temp3)
    elif option==4:
        idx = 0
        while( idx < len(memory) ):
            temp = memory[idx]
            temp2 = memory[idx+1]
            temp3 = memory[idx+2]
            ret.append(max(temp2, temp3) + temp, option=no_grad)
    elif option==5:
        idx = 0
        while( idx < len(memory) ):
            temp = memory[idx]
            temp2 = memory[idx+1]
            temp3 = memory[idx+2]
            ret.append(max(temp2, temp3, temp), option=no_grad)
    elif option==6:
        ret = memory


    return ret     

def to_cuda(t):
    if torch.cuda.is_available():
        return t.cuda()
    else:
        return t
    
