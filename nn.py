
# coding: utf-8

# In[111]:

import random
import math


# In[112]:

class Dataset:
    
    def __init__(self,filename):
        
        self.input = []
        self.output = []
        file = open(filename)
        line = file.readline()
        line = line.strip()
        line_list = line.split(",")
        self.dic = [{} for i in range(len(line_list))]
        
        while line:
            line = line.strip()
            line_list = line.split(",")
            input_line = []
            for i in range(len(line_list)-1):
                if line_list[i] not in self.dic[i]:
                    now_length = len(self.dic[i])
                    self.dic[i][line_list[i]] = now_length + 1
                input_line.append(self.dic[i][line_list[i]])
            self.input.append(input_line)
            if line_list[-1] not in self.dic[-1]:
                now_length = len(self.dic[-1])
                self.dic[-1][line_list[-1]] = now_length + 1
            self.output.append(self.dic[-1][line_list[-1]])
            line = file.readline()
            
    def show(self):
        print(self.input)
        print(self.output)
        print(self.dic)


# In[1]:

def NN(dataset,train_ratio=0.8,cross_v=1,hidden_layer_nodes=[7],learning_rate=0.1,epochs=3000,weight_path=None):
    i_num = len(dataset.dic)-1
    o_num = len(dataset.dic[-1])
    
    # construct a network
    learned_net = network(i_num,hidden_layer_nodes,o_num)
    
    dataset_size = len(dataset.input) 
    list_index = [i for i in range(dataset_size)]
    if cross_v == 1:
        train = int(len(dataset.input) * train_ratio)
    elif cross_v > 1:
        train = int(len(dataset.input) * (cross_v-1)/cross_v)
        
    train_acc = []
    test_acc = []
    
    if weight_path:
        file = open(weight_path)
        for layer_num in range(1,len(learned_net)):
            layer = learned_net[layer_num]
            for node in layer:
                line = file.readline()
                line = line.strip()
                weights_list = line.split(' ')
                node.weights = [float(weight) for weight in weights_list]
                
        predict_acc = 0
        for i in range(dataset_size):
            result = predict(dataset.input[i],learned_net)
            if result == dataset.output[i]:
                predict_acc += 1
                
        acc_ratio = predict_acc/dataset_size
        
        print("trained model accuracy: {}%".format(acc_ratio*100))
        
        return learned_net
                
    else: 
        # train the model 
        for train_time in range(cross_v):
            
            random.shuffle(list_index)
            train_data = [dataset.input[list_index[i]] for i in range(train)]
            train_class = [dataset.output[list_index[i]] for i in range(train)]
            test_data = [dataset.input[list_index[i]] for i in range(train,dataset_size)]
            test_class = [dataset.output[list_index[i]] for i in range(train,dataset_size)]
            learned_net = backpropagation(train_data,
                                          train_class,learned_net, learning_rate, epochs)

            # compute accuracy from training data
            train_predict = 0
            for i in range(train):
                result = predict(train_data[i],learned_net)
                if result == train_class[i]:
                    train_predict += 1
            train_acc_ratio = train_predict/train
            train_acc.append(train_acc_ratio)

            # compute accuracy from testing data
            test_predict = 0
            for i in range(dataset_size - train):
                result = predict(test_data[i],learned_net)
                if result == test_class[i]:
                    test_predict += 1
            test_acc_ratio = test_predict/(dataset_size - train)
            test_acc.append(test_acc_ratio)

        print("training data accuracy: {}%".format(sum(train_acc)/cross_v*100))
        print("testing data accuracy: {}%".format(sum(test_acc)/cross_v*100))

        return learned_net


# In[114]:

def new_predict(data_dic,test_data,net):
    for i in range(len(test_data)):
        test_data[i] = data_dic[i][test_data[i]]
        
    result = predict(test_data,net)
    for key in data_dic[-1]:
        if data_dic[-1][key] == result:
            print (key)
            return None


# In[115]:

def predict(test_data,net):
    #put input to the first layer
    for k in range(len(net[0])):
        net[0][k].value = test_data[k]
            
    #forward propagation and activation
    for layer in net[1:]:
        for node in layer:
            inc = [n.value for n in node.inputs]
            in_val = dotproduct(inc,node.weights)
            node.value = sigmoid(in_val)       
    
    max_num = net[-1][0].value
    result_class = 0
    for i in range(len(net[-1])):
        if net[-1][i].value > max_num:
            max_num = net[-1][i].value
            result_class = i
    
    return (result_class+1)   


# In[116]:

def random_weights(min_v, max_v, num):
    return [random.uniform(min_v,max_v) for i in range(num)]


# In[117]:

def dotproduct(x_list,y_list):
    return sum(x * y for x, y in zip(x_list, y_list))


# In[118]:

def sigmoid(num):
    return 1/(1 + math.exp(-num))


# In[119]:

def sigmoid_d(num):
    return num * (1 - num)


# In[120]:

def backpropagation(d_input, d_output, net, learning_rate, epochs):
    
    for layer in net:
        for node in layer:
            node.weights = random_weights(-0.5,0.5,len(node.weights))
            
    data_input = d_input
    data_output = []
    
    for i in range(len(d_output)):
        t = [0 for i in range(len(net[-1]))]
        index = d_output[i]-1
        t[index] = 1
        data_output.append(t)
            
    for epoch in range(epochs):
        
        if epoch % 500 == 0:
            print("training step: %d/%d"%(epoch,epochs))
        
        for i in range(len(data_input)):
            
            i_val = data_input[i]
            o_val = data_output[i]
            
            #put input to the first layer
            for k in range(len(net[0])):
                net[0][k].value = i_val[k]
            
            #forward propagation and activation
            for layer in net[1:]:
                for node in layer:
                    inc = [n.value for n in node.inputs]
                    in_val = dotproduct(inc,node.weights)
                    node.value = sigmoid(in_val)           
            
            delta = [[] for k in range(len(net))]
            
            
            err = [(o_val[k] - net[-1][k].value) for k in range(len(net[-1]))]
            
            delta[-1] = [(sigmoid_d(net[-1][k].value) * err[k]) for k in range(len(net[-1]))]
            
            # back propagation
            for j in range(len(net)-2,0,-1):
                layer = net[j]
                nx_layer = net[j+1]       
                w = [[node.weights[k] for node in nx_layer] for k in range(len(layer))]           
                delta[j] = [sigmoid_d(layer[k].value) * dotproduct(w[k], delta[j+1]) for k in range(len(net[j]))]
                
            # update weights
            for j in range(1, len(net)):
                layer = net[j]
                inc = [node.value for node in net[j-1]]
                for node_index in range(len(net[j])):
                    node = net[j][node_index]
                    for index in range(len(net[j-1])):
                        node.weights[index] = node.weights[index]+learning_rate*delta[j][node_index]*inc[index]
    
    return net    


# In[121]:

class NN_nodes:
    
    def __init__(self):
        self.weights = []
        self.inputs = []
        self.value = None


# In[122]:

def network(i_num, hidden_layer_nodes, o_num):
    
    if len(hidden_layer_nodes) == 1 and hidden_layer_nodes[0] == 0:
        total_layer = [i_num] + [o_num]
    else:
        total_layer = [i_num] + hidden_layer_nodes + [o_num]
    
    print('model layers:', total_layer)
        
    net = [[NN_nodes() for i in range(size)] for size in total_layer]
    
    for i in range(1,len(net),1):
        for node in net[i]:
            for prev in net[i-1]:
                node.inputs.append(prev)
                node.weights.append([0])
                
    return net


# In[124]:

import argparse

parser = argparse.ArgumentParser(description='NN')
parser.add_argument('-a','--alpha', type=float, default=0.1)
parser.add_argument('-d','--datapath', default='iris.discrete.txt')
parser.add_argument('-e','--epoch',type=int, default=1000)
parser.add_argument('-hl','--hidden_layers', default='7')
parser.add_argument('-c','--crossvalidation',type=int,default=5)
parser.add_argument('-w','--weightspath',default=None)

args = parser.parse_args()
learning_rate = args.alpha
datapath = args.datapath
epoch = args.epoch
layers = args.hidden_layers
cross_v = args.crossvalidation
weight_path = args.weightspath

dataset = Dataset(datapath)

hidden_layers = layers.split('-')
hidden_layers = [int(number) for number in hidden_layers]

learned_net = NN(dataset,cross_v=cross_v,hidden_layer_nodes=hidden_layers,learning_rate=learning_rate,epochs=epoch,weight_path=weight_path)

print('Do you want to predict? Please type the attributes (END means exit):')
line = input()
while not line.startswith('END'):
    line = line.strip()
    test_data = line.split(' ')
    if (len(test_data)) != len(dataset.input[0]):
        test_data = line.split(',')
    new_predict(dataset.dic, test_data, learned_net)
    print('Do you want to predict? Please type the attributes (END means exit):')
    line = input()

