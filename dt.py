
# coding: utf-8

# In[258]:

from math import log
import argparse


# In[259]:

def load_data(filepath):
    file = open(filepath)
    line = file.readline()
    line = line.strip()
    labels = line.split(',')
    data_set = []
    line = file.readline()
    class_value = []
    
    while line:
        line = line.strip()
        line_list = line.split(',')
        data_set.append(line_list)
        line = file.readline()
        if line_list[-1] not in class_value:
            class_value.append(line_list[-1])
        
    return data_set,labels[:-1],class_value


# In[260]:

def predict(model_dic, labels, class_values, test_data):
    
    key = list(model_dic.keys())[0]
    index_number = labels.index(key)
    
    while model_dic[key][test_data[index_number]] not in class_value:
        model_dic = model_dic[key][test_data[index_number]]
        key = list(model_dic.keys())[0]
        index_number = labels.index(key)
    
    return model_dic[key][test_data[index_number]]


# In[261]:

def majority(classList):
    classcount={}
    for class_v in classList:
        if class_v not in classcount.keys(): 
            classcount[class_v] = 0
        classcount[class_v] += 1
    
    max_number = -1
    class_value = None
    for key in classcount.keys():
        if classcount[key] > max_number:
            max_number = classcount[key]
            class_value = key
    
    return class_value


# In[262]:

def choose_attribute(data):
    attributes = len(data[0]) - 1
    baseEntropy = entropy(data)
    max_InfoGain = 0.0;
    bestattr = -1
    for i in range(attributes):
        featList = [d[i] for d in data]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            newData = split(data, i, value)
            probability = len(newData)/float(len(data))
            newEntropy += probability * entropy(newData)
        infoGain = baseEntropy - newEntropy
        if infoGain > max_InfoGain:
            max_InfoGain = infoGain
            bestattr = i
    return bestattr


# In[263]:

def entropy(data):
    entries = len(data)
    labels = {}
    for data_list in data:
        label = data_list[-1]
        if label not in labels.keys():
            labels[label] = 0
        labels[label] += 1
    entropy = 0.0
    for key in labels:
        probability = float(labels[key])/entries
        entropy -= probability * log(probability,2)
    return entropy


# In[264]:

def split(data, index, val):
    newData = []
    for data_list in data:
        if data_list[index] == val:
            reduced = data_list[:index]
            reduced.extend(data_list[index+1:])
            newData.append(reduced)
    return newData


# In[265]:

def tree(data,labels):
    classList = [d[-1] for d in data]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # if there is no more attribute, return the majority of remaining classes 
    if len(data[0]) == 1:
        return majority(classList)
    # choose the best attribute
    bestFeat = choose_attribute(data)
    bestFeatLabel = labels[bestFeat]
    theTree = {bestFeatLabel:{}}
    #del(labels[bestFeat])
    labels = labels[:bestFeat] + labels[bestFeat+1:]
    featValues = [d[bestFeat] for d in data]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        theTree[bestFeatLabel][value] = tree(split(data, bestFeat, value),subLabels)
    return theTree


# In[266]:

parser = argparse.ArgumentParser(description='DT')
parser.add_argument('-d','--datapath', default='iris.data.txt')
args = parser.parse_args()
datapath = args.datapath
data_set, labels, class_value = load_data(datapath)
model = tree(data_set,labels)
print(model)
right_number = 0
for data_list in data_set:
    result = predict(model, labels, class_value, data_list[:-1])
    if result == data_list[-1]:
        right_number += 1     
print('accuracy: {}%'.format(right_number/len(data_set)*100))


# In[267]:

print('Do you want to predict? Please type the attributes (END means exit):')
line = input()
while not line.startswith('END'):
    line = line.strip()
    test_data = line.split(' ')
    if len(test_data) != len(labels):
        test_data = line.split(',')
    result = predict(model, labels, class_value, test_data)
    print(result)
    print('Do you want to predict? Please type the attributes (END means exit):')
    line = input()


# In[ ]:



