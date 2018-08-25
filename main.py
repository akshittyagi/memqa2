import os
import pickle as pkl

import torch
import torch.nn as nn
from torch.autograd import Variable
from gensim.models import KeyedVectors

from memnn import Network
from prepareForLSTM import createTupleSentences
from train import train, test
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

def run(grad='4', mode='train'):
    
    if mode == 'train':
        train_data_path = os.path.join(dirname, elementary+'Train'+postfix)
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
        word_to_index['<unk>'] = len(word_to_index)
        pkl.dump(tupleData, open('MemoryString.pkl', 'w'))
        pkl.dump(word_to_index, open('Dict.pkl', 'w'))

        memory = prepareMemory(tupleData, word_to_index)
        pkl.dump(memory, open('MemorySent.pkl', 'w'))

        model = Network(mem_emb_size=600, vocab_size=len(word_to_index), embedding_size=600, hops=1)

        # filename = 'GoogleNews-vectors-negative300.bin'
        # modelW2V = KeyedVectors.load_word2vec_format(filename, binary=True)
        
        # list_vectors = []
        # cnt = 0
        # for word in word_to_index:
            # if word in modelW2V:
                # vector = modelW2V[word]
            # else:
                # vector = torch.zeros(300)
                # cnt += 1
            # list_vectors.append(torch.Tensor(vector))
        # del modelW2V
        # print cnt #Will have to do an infer_vector
        # embedVectors = torch.stack(list_vectors, dim=0)
        # model.A.data = embedVectors.clone()
        # model.B.data = embedVectors.clone()
        # model.C.data = embedVectors.clone()

        if torch.cuda.is_available():
            model.cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_function = nn.CrossEntropyLoss()

        train(model, training_data, 70, word_to_index, memory, loss_function, optimizer)
        torch.save(model.state_dict(), grad+"_Model")
        test_data_path = os.path.join(dirname, elementary+'Test'+postfix)
        testing_data = pkl.load(open(test_data_path, 'r'))

        testData = []
        for key, value in testing_data.iteritems():
            sentence = value
            testData.append(sentence)
        test(model, testData, word_to_index, memory, batch_size=16, pathForRelMem='relevantMemoryTest.pkl')
        
    elif mode == 'test':
        test_data_path = os.path.join(dirname, elementary+'Test'+postfix)
        tupleData = pkl.load(open('MemoryString.pkl', 'r'))
        word_to_index = pkl.load((open('Dict.pkl', 'r')))
        model = Network(mem_emb_size=600, vocab_size=len(word_to_index), embedding_size=600, hops=1)
        # model.load_state_dict(torch.load(grad+"_Model"))
        model.load_state_dict(torch.load(grad+"_Model")['state_dict'])
        memory = pkl.load(open('MemorySent.pkl', 'r'))

        testing_data = pkl.load(open(test_data_path, 'r'))

        testData = []
        for key, value in testing_data.iteritems():
            sentence = value
            testData.append(sentence)
        test(model, testData, word_to_index, memory, batch_size=16, pathForRelMem='relevantMemoryTest.pkl')
        
if __name__=="__main__":
    # run(grad='4', mode='test')
    run()
