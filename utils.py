import os
import torch
import pickle as pkl
import torch.nn as nn
    
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
    
