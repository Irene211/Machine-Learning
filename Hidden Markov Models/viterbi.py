# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:07:23 2019

@author: Yelena
"""
import numpy as np
import sys
#
traintxt = sys.argv[1]
idx_to_word = sys.argv[2]
inx_to_tag = sys.argv[3]
hmmprtxt = sys.argv[4]
hmmemtxt = sys.argv[5]
hmmtrtxt = sys.argv[6]
predtxt = sys.argv[7]
metricstxt = sys.argv[8]

index_to_word, index_to_tag,new_list = {}, {}, []  
with open(idx_to_word) as f2:
    line = f2.readlines()
    line = [i.strip() for i in line]
    for index in range(len(line)):
        index_to_word[line[index]] = index 
   
with open(inx_to_tag) as f3:
    tag = f3.readlines()
    tag = [i.strip() for i in tag]
    for index in range(len(tag)):
        index_to_tag[tag[index]] = index 

with open(hmmtrtxt) as tr:
    line = tr.readlines()
    line = [i.strip() for i in line]
    line = [i.split(" ") for i in line]
    line = [[float(i) for i in j] for j in line]
    m_trans = np.array((line))
    
with open(hmmprtxt) as pr:
    line = pr.readlines()
    line = [i.strip() for i in line]
    line = [float(i) for i in line]
    m_init = np.array(line)
    
with open(hmmemtxt) as em:
    line = em.readlines()
    line = [i.strip() for i in line]
    line = [i.split(" ") for i in line]
    line = [[float(j) for j in i] for i in line]
    m_emis = np.array(line)
        
with open(traintxt) as ft:
    line = ft.readlines()
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
    

states = [index_to_tag[i] for i in index_to_tag]
res_list,new_seq = [], []
n = 0
for ii in range(len(new_list)):  #loop sentense
    n +=1
    if n == 10001:
        break
    seq = new_list[ii]
    
    w = np.zeros((len(seq),len(index_to_tag)))
    word_tag_1 = seq[0]
    word_index_1 = int(word_tag_1.split("_")[0])
    
    for seq_len in range(len(w[0])):
       
        w[0,seq_len] = np.log(m_init[seq_len]) + np.log(m_emis[seq_len,word_index_1])
    
    p = np.zeros((len(seq),len(index_to_tag)))
    p[0,:] = np.array(states)
 
    for index in range(1, len(seq)):  #loop word
        word_tag = seq[index]
        word_index = int(word_tag.split("_")[0])
        
        
        for i in states:
            maxnum = -1000000.0
            
            for k in states:
               
                mark = np.log(m_emis[i,word_index]) + np.log(m_trans[k,i]) + w[index-1,k]
               
                if mark > maxnum:
                    marp = k
                    maxnum = mark
               
            w[index,i], p[index,i] = maxnum, marp    
   
            
    max_index = np.argmax(w[-1,:])
    index_list = []
    
    for row in p[::-1]:
        index_list.append(max_index)
        
        index_stored = row[max_index]
        
        max_index = int(index_stored)
    index_list = index_list[::-1]    

    new_word = []
    for i in range(len(seq)):
        new_word.append( seq[i][:-1] + str(index_list[i]) )
    new_seq.append(new_word) 

  
def accuracy(new_seq,new_list):
    acc = 0
    for i in range(len(new_seq)):
        for j in range(len(new_seq[i])):
            if new_seq[i][j] == new_list[i][j]:
                acc += 1
    total = [j for i in new_seq for j in i]
    res = float(acc/len(total))
    with open(metricstxt,"w") as f_metric:
        f_metric.write("Accuracy: " + str(res))
    return res
    
res_seq = ""   
for i in new_seq:
    for j in i:
        word = int(j.split("_")[0])
        tag = int(j.split("_")[1])
        res_seq += str(list(index_to_word)[word]) + "_" + str(list(index_to_tag)[tag]) + " "
    res_seq = res_seq[:-1]
    res_seq += "\n"
    
with open(predtxt,"w") as f_metric:
    f_metric.write(res_seq)


acc = accuracy(new_seq, new_list)

#
#with open("C:/Users/Yelena/Desktop/10601/601 - hw7/handout/fulldataoutputs/predictedtest.txt") as ft:
#    line1 = ft.readlines()
#
#with open("C:/Users/Yelena/Desktop/10601/601 - hw7/handout/fulldataoutputs/predicted.txt") as ft:
#    line2 = ft.readlines()
#
#line1 == line2










