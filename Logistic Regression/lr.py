# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:16:53 2019

@author: Yelena
"""
import sys
file1 = sys.argv[1]
file2 = sys.argv[2]
file3 = sys.argv[3]
dict_file = sys.argv[4]
out_file1 = sys.argv[5]
out_file3 = sys.argv[6]
out_file2 = sys.argv[7]
num_epoch = sys.argv[8]
import numpy as np

#Calculate the dot product of theta*x
#X is a dictionary(sparse_x), W is a list
def sparse_dot(X, W, bias):
    product = 0.0
    for key in X:
        key = int(key)
        product += W[key]
    return product + bias

def get_file(file):
    with open(file) as f:
        line = f.readlines()
        line = [x.strip("\n") for x in line]
        line = [x.split("\t") for x in line]
       
        label, x_list = [], []
        for paragraph in line:
            label.append(paragraph[0])
            
            x_dict = dict()
            for relationship in paragraph[1:]:
                relationship = relationship.split(":")
                x_dict[relationship[0]] = relationship[1]
            x_list.append(x_dict)
        #num_examples = len(x_list) #350
    return label, x_list

def model(num_epoch, label, x_list, dict_file, bias):
    with open(dict_file) as ff:
        txt = ff.readlines()
        len_dict = len(txt)
        theta = np.zeros(len_dict) 
    
    for i in range(0, num_epoch):
        for x_list_sub, label_sub in zip(x_list, label):
            x_ji = np.zeros(len_dict) 
            for index in x_list_sub:   
                x_ji[int(index)] = 1.0
            product = sparse_dot(x_list_sub, theta, bias)
            subtract_y = float(label_sub) - ( np.exp(product) / (1 + np.exp(product))  )
           
            bias +=  0.1 * subtract_y
            
            theta += 0.1 * x_ji * subtract_y
    return theta, bias

def predict(theta, x_list, label, out_file, bias):
    predict = ""
    for x_list_one, label_sub_one in zip(x_list, label):
        u = sparse_dot(x_list_one, theta, bias)
        product = np.exp(u) / (1 + np.exp(u)) 
      
        if product > 0.5:
            predict += "1" + "\n"
        else:
            predict += "0" + "\n"
    #Write predict res
    with open(out_file,"w") as fff:
        fff.write(predict)
    return predict

def output(predict, label):
    error = 0
    predict_labels = [x.strip("\n") for x in predict]
    predict_labels = [x for x in predict_labels if x!=""]
    for i in range(len(predict_labels)):
        if predict_labels[i] != label[i]:
            error += 1
    error_rate = error/len(predict_labels)
   
    return error_rate

def main(num_epoch, file1, file2, file3, dict_file, out_file1, out_file2,out_file3):
#    file1 = "C:/Users/Yelena/Desktop/10601/601 - hw4/handout/largeoutput/model1_formatted_train.tsv"
#    file2 = "C:/Users/Yelena/Desktop/10601/601 - hw4/handout/largeoutput/model1_formatted_valid.tsv"
#    file3 = "C:/Users/Yelena/Desktop/10601/601 - hw4/handout/largeoutput/model1_formatted_test.tsv"
#    dict_file = "C:/Users/Yelena/Desktop/10601/601 - hw4/handout/largeoutput/dict.txt"
#    out_file1 = "C:/Users/Yelena/Desktop/10601/601 - hw4/handout/largeoutput/train_out.labels"
#    out_file2 = "C:/Users/Yelena/Desktop/10601/601 - hw4/handout/largeoutput/metrics_out.txt"
#    out_file3 = "C:/Users/Yelena/Desktop/10601/601 - hw4/handout/largeoutput/test_out.labels"
    num_epoch = int(num_epoch)
    label_train, x_list_train =  get_file(file1)
    
    label_test, x_list_test =  get_file(file3)
    bias = 0
    theta, bias_train = model(num_epoch, label_train, x_list_train, dict_file, bias)
    
    predict_train = predict(theta, x_list_train, label_train, out_file1, bias_train)
    predict_test = predict(theta, x_list_test, label_test, out_file3, bias_train)
    
    error_train = output(predict_train, label_train)
    error_test = output(predict_test, label_test)
    
    with open(out_file2,"w") as fff:
            fff.write(str(error_train)+"\n")
            fff.write(str(error_test))     



main(num_epoch, file1, file2, file3, dict_file, out_file1, out_file2,out_file3)

#python lr.py model1_formatted_train.tsv model1_formatted_valid.tsv model1_formatted_test.tsv dict.txt train_out.labels test_out.labels metrics_out.txt 50

#(base) C:\Users\Yelena\Desktop>python lr.py model2_formatted_train.tsv model2_formatted_valid.tsv model2_formatted_test.tsv dict.txt train_out.labels test_out.labels metrics_out.txt 50











































































