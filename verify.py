import json 
import pickle as pkl 


def verifyDataSetFile(path):
    
    data = pkl.load(open(path, 'r'))
    count = 0
    for key, value in data.iteritems():
        print key
        print value
        count += 1
        print value
    print count


if __name__=="__main__":
    verifyDataSetFile('Omnibus-Gr04-NDMC-Train.csv_.pkl')