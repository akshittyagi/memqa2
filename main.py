import os
import pickle as pkl

import torch
import torch.nn as nn
from torch.autograd import Variable

from memnn import Network
from prepareForLSTM import createTupleSentences
from train import train
from utils import to_cuda

dirname = os.getcwd()
elementary = 'Omnibus-Gr04-NDMC-'
middle = 'ScienceQuestionsV2-Middle-NDMC-'
postfix = '.csv_.pkl'
dirKB = 'TupleInfKB'
tupleKb = 'thGradeOpenIE.txt'

def prepareMemory(memory, word_to_index):

    ret = []
    for sentence in memory:
        idxs = [word_to_index[w] for w in sentence.split()]
        ret.append(Variable(to_cuda(torch.LongTensor(idxs))))
    return ret

def run(grad='4'):
    train_data_path = os.path.join(dirname, elementary+'Train'+postfix)
    test_data_path = os.path.join(dirname, elementary+'Test'+postfix)
    KBData_path = os.path.join(dirKB, grad+tupleKb)

    training_data = pkl.load(open(train_data_path, 'r'))
    KBData = open(KBData_path, 'r')
    word_to_index = {}
    trainData = []
    for key, value in training_data.iteritems():
        sentence = value
        trainData.append(sentence)
        currSent = sentence.split('+')
        tempChoice = currSent[1].split('/')
        currSent[1] = " ".join(tempChoice)
        currSent[2] = tempChoice[ord(currSent[2])-ord('A')]
        new_sentence = " ".join(currSent)
        for word in new_sentence.split():
            if word not in word_to_index:
                word_to_index[word] = len(word_to_index)

    print "Data Curated" 

    tupleData, word_to_index, memory = createTupleSentences(dirKB,grad+tupleKb,save=False, word_to_index=word_to_index)
    print "Tuple Set Created"
    word_to_index['<pad>'] = len(word_to_index)

    pkl.dump(tupleData, open('MemoryString.pkl', 'w'))

    memory = prepareMemory(tupleData, word_to_index)
    
    model = Network(mem_emb_size=300, vocab_size=len(word_to_index), embedding_size=300, hops=1)
    if torch.cuda.is_available():
        model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=4e-4)
    loss_function = nn.CrossEntropyLoss()

    train(model, trainData, 5, word_to_index, memory, loss_function, optimizer)


if __name__=="__main__":
    run()
