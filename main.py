import os
import pickle as pkl

import torch
import torch.nn as nn

from memnn import Network
from prepareForLSTM import createTupleSentences
from train import train
dirname = os.getcwd()
elementary = 'Omnibus-Gr04-NDMC-'
middle = 'ScienceQuestionsV2-Middle-NDMC-'
postfix = '.csv_.pkl'
dirKB = 'TupleInfKB'
tupleKb = 'thGradeOpenIE.txt'

def run(grad='4'):
    train_data_path = os.path.join(dirname, elementary+'Train'+postfix)
    test_data_path = os.path.join(dirname, elementary+'Test'+postfix)
    KBData_path = os.path.join(dirKB, grad+tupleKb)

    print train_data_path
    training_data = open(train_data_path, 'r')
    KBData = open(KBData_path, 'r')
    word_to_index = {}
    trainData = []
    for sentence in training_data:
        trainData.append(sentence)
        currSent = sentence.split('+')
        print currSent
        tempChoice = currSent[1].split('/')
        currSent[1] = " ".join(tempChoice)
        currSent[2] = tempChoice[ord(currSent[1])-ord('A')]
        new_sentence = " ".join(currSent)
        for word in new_sentence.split():
            if word not in word_to_index:
                word_to_index[word] = len(word_to_index)
    
    tupleData, word_to_index = createTupleSentences(dirKB,grad+tupleKb,save=False, word_to_index=word_to_index)
   
    model = Network(vocab_size=len(word_to_index), embedding_size=300, hops=1)
    if torch.cuda.is_available():
        model.cuda()
    optimizer = torch.optim.SGD(model.parameters())
    loss_function = nn.CrossEntropyLoss()
    train(model, trainData, 10, word_to_index, memory, loss_function, optimizer)


if __name__=="__main__":
    run()
