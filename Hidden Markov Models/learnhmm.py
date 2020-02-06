# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 00:36:49 2019

@author: Yelena
"""
import numpy as np
import sys
train_input = sys.argv[1]
idx_to_word = sys.argv[2]
idx_to_tag = sys.argv[3]
hmmpri = sys.argv[4]
hmmem = sys.argv[5]
hmmtr = sys.argv[6]


def parse_file(file1, file2, file3):
    index_to_word, index_to_tag,new_list = {}, {}, []  
    with open(file1) as f2:
        line = f2.readlines()
        line = [i.strip() for i in line]
        for index in range(len(line)):
            index_to_word[line[index]] = index 
       
    with open(file2) as f3:
        tag = f3.readlines()
        tag = [i.strip() for i in tag]
        for index in range(len(tag)):
            index_to_tag[tag[index]] = index 
    
    
    with open(file3) as f:
        line = f.readlines()
        line = [i.strip() for i in line]
        line = [i.split(" ") for i in line]
        for line_list in line:
            sub_new_list = []
            for word_tag in line_list:
                word, tag = word_tag.split("_")[0], word_tag.split("_")[1]
                word_index, tag_index = index_to_word[word], index_to_tag[tag]
                new_word_tag = str(word_index) + "_" + str(tag_index)
                sub_new_list.append(new_word_tag)
            new_list.append(sub_new_list)
        new_list = new_list[:10]
    return new_list,index_to_tag,index_to_word

def p_init(new_list,index_to_tag):
    length = len(index_to_tag)
    p_init = np.zeros(length)
    for i in new_list:
        for key in index_to_tag:
            if i[0][-1] == str(index_to_tag[key]):
               
                p_init[index_to_tag[key]] += 1.0
            
    p_init += 1.0
    p_init  /= p_init.sum()
    return p_init
   
def p_trans(new_list,index_to_tag):
    length = len(index_to_tag)
    matrix = np.zeros((length, length))

    for i in range(len(new_list)):
        for j in range(len(new_list[i])-1):
            tag = new_list[i][j][-1]
            next_tag = new_list[i][j+1][-1]
      
            matrix[int(tag)][int(next_tag)] += 1.0
        
    matrix += 1.0
    matrix /= matrix.sum(axis=1)[:,None]
    return matrix
    
def p_emission(new_list, index_to_tag, index_to_word):
    tag_len = len(index_to_tag)
    word_len = len(index_to_word)
    matrix = np.zeros((tag_len, word_len))
    for i in range(len(new_list)):
        for j in range(len(new_list[i])):
            word, tag = new_list[i][j].split("_")[0], new_list[i][j].split("_")[1]
            matrix[int(tag)][int(word)] += 1
        
    matrix += 1.0
    matrix /= matrix.sum(axis=1)[:,None]
    return matrix

def main(file_1, file_2, file_3):
    new_list,index_to_tag,index_to_word = parse_file(file_1, file_2, file_3)
    #new_list,index_to_tag,index_to_word = parse_file("toy_index_to_word.txt","toy_index_to_tag.txt","toytrain.txt")
    matrix_init = p_init(new_list, index_to_tag) 
    matrix_trans = p_trans(new_list, index_to_tag)
    matrix_emission = p_emission(new_list, index_to_tag, index_to_word)
    return matrix_init, matrix_trans, matrix_emission,index_to_tag, index_to_word,new_list
    
    
matrix_init, matrix_trans, matrix_emission,index_to_tag, index_to_word,new_list = main(idx_to_word,idx_to_tag,train_input)
#matrix_init, matrix_trans, matrix_emission,index_to_tag, index_to_word,new_list = main("index_to_word.txt","index_to_tag.txt","trainwords.txt")
matrix_emission = matrix_emission.tolist()
matrix_trans = matrix_trans.tolist()
matrix_init = matrix_init.tolist()
init_str, em_str, tr_str = "", "", ""


for i in matrix_emission:
    for j in i:
        em_str += str(j) + " "
    em_str += "\n"   


with open(hmmpri,"w") as w1:
    for i in matrix_init:
        init_str += str(i) + "\n"
    w1.write(init_str)


with open(hmmem,"w") as w2:
    w2.write(em_str)
      

with open(hmmtr,"w") as w3:    
    for i in matrix_trans:
        for j in i:
            tr_str += str(j) + " "
        tr_str += "\n"
    w3.write(tr_str)




    
    
    
    
    
    
    
    
    
    
    
    
    
    






