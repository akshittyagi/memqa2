import json 
import pickle as pkl 


def verifyDataSetFile(path):
    
    data = pkl.load(open(path, 'r'))
    count = 0
    for key, value in data.iteritems():
        print key
        print value
        count += 1

    print count


if __name__=="__main__":
    verifyDataSetFile('/Users/akshittyagi/memqa/Omnibus-Gr04-NDMC-Test.csv_FORMATTED.pkl')