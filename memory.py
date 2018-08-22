import pickle
import json
import os
import ipdb
import torchtext
import numpy as np

def loadDataset(dataTorchFp):
    value = torchtext.data.Field(sequential=True, use_vocab=True)
    key = torchtext.data.Field(sequential=False, use_vocab=True)
    torchData = torchtext.data.TabularDataset(path=dataTorchFp, format='json', \
            fields={'key':('key', key), 'value':('value', value)})
    # ipdb.set_trace()
    value.build_vocab(torchData, vectors='fasttext.en.300d')
    key.build_vocab(torchData)

    return torchData

def getVectors(torchData, outputFp):
    stoi = torchData.fields['value'].vocab.stoi
    vectors = np.array(torchData.fields['value'].vocab.vectors)

    vectorsData = dict()
    for ex in torchData.examples:
        indices = [stoi[word] for word in ex.value]
        sumVector = vectors[indices].sum(axis=0)
        norm = np.linalg.norm(sumVector) 
        if(norm != 0):
            normVector = sumVector / norm 
        else: 
            normVector = sumVector
        vectorsData[ex.key] = normVector

    pickle.dump(vectorsData, open(outputFp, 'w'))
    return 

def convertToTorchText(dataFp, torchFp, Type='dict'):
    torchF = open(torchFp, 'w')
    data = pickle.load(open(dataFp, 'r'))
    for key in data:
        if(Type == 'dict'):
            value = data[key]
        elif(Type == 'list'):
            value = key
        record = dict()
        record['value'] = value 
        record['key'] = key 
        json.dump(record, torchF)
        torchF.write('\n')

    torchF.close()
    return 

def cleanTrainingData(dataFp, cleanDataFp):
    data = pickle.load(open(dataFp, 'r'))
    clean_data = dict()
    for key in data:
        value = data[key]
        fields = value.split('+')
        question = fields[0]
        answers = " ".join(fields[1].split('/'))
        clean_value = question+" "+answers
        clean_data[key] = clean_value 

    pickle.dump(clean_data, open(cleanDataFp, 'w'))
    return 

def relevantMemory(memoryVecFp, trainDataVecFp, outFp):
    trainData = pickle.load(open(trainDataVecFp, 'r'))
    memoryData = pickle.load(open(memoryVecFp, 'r'))

    memoryVec = np.array(memoryData.values())
    trainVec = np.array(trainData.values())

    scores = np.matmul(trainVec, memoryVec.T)
    bestIndices = scores.argsort()[:, -50:]

    memoryKeys = list(memoryData.keys())
    trainKeys = list(trainData.keys())

    out = dict()
    for i in range(len(trainKeys)):
        question = trainKeys[i]
        relMem = list()
        for j in range(len(bestIndices[i])):
            mem = memoryKeys[bestIndices[i][j]]
            relMem.append(mem)
        out[question] = relMem

    pickle.dump(out, open(outFp, 'w'))
    return 

def main():
    memoryFp = 'MemoryString.pkl'
    dataFp = 'Omnibus-Gr04-NDMC-Train.csv_.pkl'

    print('Getting vectors of memory...')
    torchFp = memoryFp + ".torch_"
    vectorsFp = memoryFp + ".vectors_"
    convertToTorchText(memoryFp, torchFp,Type='list')
    data = loadDataset(torchFp)
    getVectors(data, vectorsFp)
    
    print('Getting vectors of data...')
    cleanDataFp = dataFp + ".clean_"
    torchDataFp = dataFp + ".torch_"
    vectorsDataFp = dataFp + ".vectors_"
    cleanTrainingData(dataFp, cleanDataFp)
    convertToTorchText(cleanDataFp, torchDataFp, Type='dict')
    data = loadDataset(torchDataFp)
    getVectors(data, vectorsDataFp) 

    print('Getting relevant memory...')
    memoryVecFp = memoryFp + ".vectors_"
    dataVecFp = dataFp + ".vectors_"
    outFp = "relevantMemory.pkl"
    relevantMemory(memoryVecFp, dataVecFp, outFp)

    return

if __name__ == "__main__":
    main()
