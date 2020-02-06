# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 13:38:08 2019

@author: Yelena
"""
import numpy as np
import sys
import matplotlib.pyplot as plt
#train_file = sys.argv[1]
#test_file = sys.argv[2]
#train_out = sys.argv[3]
#test_out = sys.argv[4]
#metrics_out = sys.argv[5]
#num_epoch = sys.argv[6]
#hidden_units = sys.argv[7]
#init_flag = sys.argv[8]
#learning_rate = sys.argv[9]

def parse_data(train_file):
    train_data,train_label = [],[]
    with open(train_file) as f:
        line = f.readlines()
        line = [i.strip() for i in line]
        for i in range(len(line)):
            line[i] = line[i].split(",")
            train_data.append(line[i][1:])
            train_label.append(line[i][0])
        train_label = [int(i) for i in train_label]
        train_data = [[int(digit) for digit in i] for i in train_data]
    for i in train_data:
        i.insert(0,1)
    return train_data, train_label

#if init_flag==1 initialize your weights randomly from a uniform distribution over the range [-0.1,0.1] 
#if init_flag==2 initialize all weights to zero 
#For both settings, always initialize bias terms to zero.
#<hidden units>: positive integer specifying the number of hidden units.
def initialize(hidden_units,M,K, init_flag):
    # M = the number of input features and K the number of outputs.
    if init_flag == 1:
        alpha = np.random.uniform(-0.1,0.1,(hidden_units, M+1))  #4，129
        beta = np.random.uniform(-0.1,0.1,(K,hidden_units+1))   #10，5
        alpha[:,0] = 0.0
        beta[:,0] = 0.0
    elif init_flag == 2:
        alpha = np.zeros((hidden_units, M+1)) #4 rows, 129 cols
        beta = np.zeros((K,hidden_units+1)) #10 rows，5
    return alpha, beta
    
#activation at the first (hidden) layer:
def activation(a):
    return 1/(1+np.exp(-a))  

def procedureSGD(train_data, train_label, test_data, test_label,alpha, beta, learning_rate,num_epoch, metrics_out,modeltrain_out,modeltest_out):
    with open(metrics_out,"a") as f:
        f.seek(0)
        f.truncate()   
    plot_sum ,plot_train, plot_sum2,plot_test = 0, 0,0,0
    len_train, len_test = 0,0
    for e in range(1, num_epoch+1):
        sum1, sum2 , error_train, error_test= 0, 0,0,0
        for i in range(len(train_data)):
            x = np.array(train_data[i])
            y = np.zeros(10)
            y[train_label[i]] = 1
            x, a, b, z, y_, J = nn_forward(x, y, alpha, beta)
            g_alpha, g_beta = nn_backward(x,y,alpha, beta, y_, z)
            alpha = alpha - learning_rate * g_alpha
            beta = beta - learning_rate * g_beta
            
        for i in range(len(train_data)):
            x1 = np.array(train_data[i])
            y1 = np.zeros(10)
            y1[train_label[i]] = 1
            x_t, a2, b2, z2, y_2, J1 = nn_forward(x1, y1, alpha, beta)
            sum1 += J1
            plot_sum += J1
            
        for i in range(len(test_data)):
            x2 = np.array(test_data[i])
            y2 = np.zeros(10)
            y2[test_label[i]] = 1
            x_t, a2, b2, z2, y_2, J2 = nn_forward(x2, y2, alpha, beta)
            sum2 += J2
            plot_sum2 += J2
            
        train_entropy = sum1/len(train_data) 
        test_entropy = sum2/len(test_data)
        
        with open(metrics_out,"a") as f:
            f.write("epoch=" +str(e)+" crossentropy(train): " +str(train_entropy)+ "\n" + "epoch=" +str(e)\
                  + " crossentropy(test): " +str(test_entropy)+ "\n" )
        
        len_train += len(train_data) 
        len_test += len(test_data) 
    ce_train = plot_sum/len_train
    ce_test = plot_sum2/len_test
        
    #pred_train = procedurePredict(train_data, train_label, alpha, beta,modeltrain_out)
    #pred_test = procedurePredict(test_data, test_label, alpha, beta,modeltest_out)
#    for i in range(len(pred_train)):
#        if pred_train[i] != train_label[i]:
#            error_train += 1
#    for i in range(len(pred_test)):
#        if pred_test[i] != test_label[i]:
#            error_test += 1
#    with open(metrics_out,"a") as f:
#        f.write("error(train): " + str(error_train/len(train_label)) + "\n" + "error(test): " + str(error_test/len(test_label)))

    return alpha, beta,ce_train,ce_test


def procedurePredict(train_data, train_label, alpha, beta,modeltrain_out):
    res = []
    output = ""
    for i in range(len(train_data)):
        x = np.array(train_data[i])
        y = np.zeros(10)
        y[train_label[i]] = 1
        x_t, a, b, z, y_, J = nn_forward(x, y, alpha, beta)
        pred = list(y_)
        pred = pred.index(max(pred))
        res.append(pred)
   
    for i in range(len(res)):
        output += str(res[i]) + "\n"
   
    with open(modeltrain_out,"w") as f:
        f.write(output)
    return res

def sigmoid_forward(a):
    z = np.array([1])    
    res = [activation(i) for i in a]
    res = np.array(res)
    res = np.append(z, res)
    return res

def sigmoid_backward(dl,z):
    dz = z * (1-z)
    dz = np.delete(dz, 0,axis=0)
    return dl * dz

def cross_entropy_forward(y, y_):
    loss = 0.0
    for i in range(len(y)):
        loss -= y[i] * np.log(y_[i])
    return loss

def cross_entropy_backward(y,y_):
    gb = []
    gb = np.array([float(y_[i] - y[i]) for i in range(len(y))])
    gb.resize(len(gb), 1)
    return gb
  
def softmax_forward(b):
    numerator = []
    Denominator = 0
    res = []
    numerator = [np.exp(i) for i in b]
    for i in numerator:
        Denominator += i
    res = [i/Denominator for i in numerator]
    res = np.array(res)
    return res

def nn_forward(x, y, alpha, beta):
    a = alpha@x
    z = sigmoid_forward(a)
    b = beta@z.T
    y_ = softmax_forward(b)
    J = cross_entropy_forward(y, y_)
   
    a.resize(len(a), 1)
    z.resize(len(z), 1)
    b.resize(len(b), 1)
    y_.resize(len(y_), 1)
    x.resize(len(x), 1)
  
    return x, a, b, z, y_, J

def nn_backward(x, y, alpha, beta, y_, z):
    gb = cross_entropy_backward(y, y_)
    g_beta =  gb@z.T
    beta_ = np.delete(beta.T, 0, axis = 0)
    gz = beta_@gb
    ga = sigmoid_backward(gz,z)
    g_alpha = ga@x.T
    return g_alpha, g_beta


def finite_diff(x,y,theta):
    epsilon = 1e-5
    #grad = zero_vector(theta.length)
    grad_a = np.zeros(theta.shape)
    grad_b = np.zeros(theta.shape)
    
    for m in range(len(theta)):
        d = zero_vector(theta.length)
        d[m] = 1
        v = forward(x,y,theta + epsilon * d)
        v -= forward(x,y,theta-epsilon * d)
        v /= 2*epsilon
        grad[m] = v
      
train_file    ="C:/Users/Yelena/Desktop/10601/601 - hw5/handout/largeTrain.csv"
test_file ="C:/Users/Yelena/Desktop/10601/601 - hw5/handout/largeValidation.csv"
train_out = "C:/Users/Yelena/Desktop/10601/601 - hw5/handout/modeltrain_out.labels"
test_out = "C:/Users/Yelena/Desktop/10601/601 - hw5/handout/modeltest_out.labels"
metrics_out = "C:/Users/Yelena/Desktop/10601/601 - hw5/handout/metrics_out.txt"
train_data, train_label = parse_data(train_file)    
val_data, val_label = parse_data(test_file)      
learning_rate = 0.01
num_epoch = 100
#hidden_units = 50
init_flag = 1
hidden_list = [5,20,50,100,200]
global pl_train, pl_test
pl_train, pl_test = [], []

for hidden_unit in hidden_list:
    alpha, beta = initialize(int(hidden_unit),128,10, int(init_flag))          
    alpha, beta, ce_train, ce_test = procedureSGD(train_data, train_label, val_data,val_label, alpha, beta, float(learning_rate),\
                               int(num_epoch),metrics_out,train_out,test_out)    
    pl_train.append(ce_train)
    pl_test.append(ce_test)

plt.plot(hidden_list, pl_train, label ="train" )
plt.xlabel('Number of Hidden Units')

plt.ylabel('average training cross_entropy')
plt.legend()

plt.plot(hidden_list, pl_test, label = "test")
plt.xlabel('Number of Hidden Units')
plt.ylabel('average training cross_entropy')
plt.legend()        
        