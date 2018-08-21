import os
import json
import csv
import pickle as pkl

path = 'ScienceQuestionsV2-Middle-NDMC-Test.csv'

def parseQuestionsAndChoices(ques):

    idx = 0
    optionChoices = ['A','B','C','D','a','b','c','d']
    choices = ""
    qu = ""
    while(idx<len(ques)-2):
        ch1 = ques[idx]
        ch2 = ques[idx+1]
        ch3 = ques[idx+2]

        if(ch1 == '(' and ch2 in optionChoices and ch3==')'):
            temp = idx + 3
            choice = ""
            while(temp<len(ques) and ques[temp]!='('):
                choice += ques[temp]
                temp += 1
            
            idx = temp
            choice = choice.lower()
            choice = choice.strip()
            choices += choice
            choices += '/'
        else:
            qu += ques[idx]
            idx += 1
    
    return qu, choices[:-1]

count = 0

data = {}
with open(path) as fil:
    reader = csv.reader(fil, delimiter=',')
    for row in reader:
        if count == 0:
            count += 1
            continue
        ques = row[3]
        answer = row[4]
        count += 1
        ques, choices = parseQuestionsAndChoices(ques)
        data[row[0]] = ques.lower().strip() + '+' + choices + '+' + answer
        print ques.lower().strip() + '+' + choices + '+' + answer

pkl.dump(data, open(path+'_.pkl', 'w'))

print count