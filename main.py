import os
import pickle as pkl

import torch
import torch.nn as nn
from torch.autograd import Variable
from gensim.models import KeyedVectors

from memnn import Network
from prepareForLSTM import createTupleSentences, combinedTuples
from train import train, test
from utils import to_cuda, buildDictionary, getWordsForm, getWordsTuple

import ipdb
import argparse

dirname = os.getcwd()
elementary = 'Omnibus-Gr04-NDMC-'
middle = 'ScienceQuestionsV2-Middle-NDMC-'
postfix = '.csv_.pkl'
dirKB = 'TupleInfKB'
tupleKb = 'thGradeOpenIE.txt'
expDir = ""

def run(args):
    
    grad = args.grade
    if args.mode == 'train':
        train_data_path = os.path.join(dirname, elementary+'Train'+postfix)
        KBData_path = os.path.join(dirKB, grad+tupleKb)

        print('Loading training data...')
        training_data = pkl.load(open(train_data_path, 'r'))
        KBData = open(KBData_path, 'r')
        word_to_index = {}
        trainData = [training_data[key] for key in training_data]
        trainDataSentences = [" ".join(getWordsForm(formSentence)) for formSentence in trainData] 

        # for key, value in training_data.iteritems():
            # sentence = value
            # trainData.append(sentence)
            # currSent = sentence.split('+')
            # tempChoice = currSent[1].split('/')
            # currSent[1] = " ".join(tempChoice)
            # currSent[2] = tempChoice[ord(currSent[2])-ord('A')]
            # new_sentence = " ".join(currSent)
            # for word in new_sentence.split():
                # if word not in word_to_index:
                    # word_to_index[word] = len(word_to_index)

        print('Loading tuples...')
        trainTupleData = combinedTuples('Omnibus4_Combined_Tuples.txt', 'Omnibus-Gr04-NDMC-Train.csv_.pkl')
        trainTupleSentences = [" ".join(getWordsTuple(trainTupleData[key])) for key in trainTupleData]
        devTupleData = combinedTuples('Omnibus4_Combined_Tuples.txt', 'Omnibus-Gr04-NDMC-Test.csv_.pkl')

        print('Forming vocab_dictionary...')
        word_to_index = buildDictionary(trainDataSentences + trainTupleSentences)

        # tupleData, word_to_index, memory = createTupleSentences(dirKB,grad+tupleKb,save=False, word_to_index=word_to_index)

        # pkl.dump(tupleData, open('MemoryString.pkl', 'w'))
        if(not os.path.exists('Dict.pkl')):
            pkl.dump(word_to_index, open('Dict.pkl', 'w'))

        # trainMemory = prepareMemory(trainTupleData, word_to_index)
        # devMemory = prepareMemory(devTupleData, word_to_index)
        # pkl.dump(memory, open('MemorySent.pkl', 'w'))

        model = Network(mem_emb_size=300, vocab_size=len(word_to_index), embedding_size=300, hops=args.hops)

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
        
        if(args.optim == 'sgd'):
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        elif(args.optim == 'adam'):
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        loss_function = nn.CrossEntropyLoss()

        test_data_path = os.path.join(dirname, elementary+'Test'+postfix)
        dev_data = pkl.load(open(test_data_path, 'r'))

        train(model, training_data, dev_data, args.epochs, word_to_index, trainTupleData, devTupleData, loss_function, optimizer, args=args)

    elif args.mode == 'test':

        print('Running in test mode...')
        test_data_path = os.path.join(dirname, elementary+'Test'+postfix)
        print('Loading tupleData...')
        tupleData = pkl.load(open('MemoryString.pkl', 'r'))
        print('Loading vocabulary dictionary...')
        word_to_index = pkl.load((open('Dict.pkl', 'r')))
        print('Initializing model...')
        model = Network(mem_emb_size=300, vocab_size=len(word_to_index), embedding_size=300, hops=args.hops)
        # model.load_state_dict(torch.load(grad+"_Model"))
        load_model_fp = os.path.join(args.exp_dir, grad+"_Model")
        print('Loading model %s...'%load_model_fp)
        model.load_state_dict(torch.load(load_model_fp)['state_dict'])
        if(torch.cuda.is_available()):
            model.cuda()
        print('Loading entire memory...')
        memory = pkl.load(open('MemorySent.pkl', 'r'))

        print('Loading test data %s'%test_data_path)
        testing_data = pkl.load(open(test_data_path, 'r'))

        test(model, testing_data, word_to_index, memory, batch_size=16, pathForRelMem='relevantMemoryTest.pkl')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help="Operate model in this mode",
                        type=str)
    parser.add_argument('--grade', help="Questions grade", default='4',
                        type=str)
    parser.add_argument('--lr', help="Learning rate", default=0.001,
                        type=float)
    parser.add_argument('--optim', help="Optimizer to use", default='adam',
                        type=str)
    parser.add_argument('--debug', help="Operate model in debug mode",
                        action="store_true")
    parser.add_argument('--exp_dir', help="Place where all experiment related data are stored",
                        type=str)
    parser.add_argument('--hops', help="Number of hops in memory", default=1,
                        type=int)
    parser.add_argument('--epochs', help="Number of epochs", default=10,
                        type=int)
    parser.add_argument('--dropout', help="Dropout value", default=0.0,
                        type=float)

    return parser

if __name__=="__main__":
    parser = parse_args()
    args = parser.parse_args()
    args.exp_dir = 'exp/'+args.exp_dir
    if(not os.path.exists(args.exp_dir)):
        os.makedirs(args.exp_dir)

    run(args)
