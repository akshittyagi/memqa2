import os
import torch
import string
import re
import pickle as pkl

pattern = re.compile("[a-zA-Z]")

def prepare(sentence, toIndex):
    idxs = [toIndex[w] for w in sentence]
    return torch.LongTensor(idx)

def createTupleSentences(dirname, filename, save=False, word_to_index = {}):
    print "Memory before tuples", len(word_to_index)
    memory = []
    path = os.path.join(dirname, filename)
    trainingData = []
    fil = open(path, 'r')
    for line in fil:
        if len(line) > 1:
            currLine = line.split()
            if len(currLine)>1:
                if "(" in currLine[1] and ")" in currLine[1]:
                    Tuple = currLine[1]
                    Tuple = Tuple[1:-1]
                    Tuple = Tuple.split(';')
                    memory.append(Tuple)

            currString = line.translate(None, string.punctuation)
            if currString is not " ":
                currString = currString.split()
                idx = 0
                while(idx<len(currString) and (pattern.match(currString[idx]) is None)):
                    idx += 1
                if idx == len(currString):
                    continue
                currString = currString[idx:]
                currString = " ".join(currString)
                trainingData.append(currString)
                
    for sentence in trainingData:
        currSent = sentence.split()
        for word in currSent:
            if word not in word_to_index:
                word_to_index[word] = len(word_to_index)

    print "Vocabulary Size", len(word_to_index)
    print "Lines", len(trainingData)

    if save:
        pkl.dump(trainingData, open(path+'_TKB.pkl', 'w'))
        pkl.dump(word_to_index, open(path+'_TKBDICT.pkl', 'w'))
    return trainingData, word_to_index, memory

if __name__ == '__main__':
    createTupleSentences('TupleInfKB', '4thGradeOpenIE.txt', save=True)
