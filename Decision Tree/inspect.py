# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:43:21 2019

@author: Yelena
"""
import sys
import math

assert len(sys.argv) == 3
train_file = sys.argv[1]
outfile = sys.argv[2]
def read_parse(file):
    with open(file) as f:
        content = f.readlines()
        content = [x.strip("\r\n") for x in content]
        content = [x.strip("\n") for x in content]
        content = [x.strip(" ") for x in content]
        attr_name = content[0].split("\t")
        party_index = len(attr_name) - 1
        array = [[party for party in line.split("\t")] for line in content[1:]]
        party = [line[party_index] for line in array]
        
        correct = ["democrat","y","A","before1950","yes","morethan3min","fast","expensive","high","Two","large"]

        correct_class = [x for x in party if x in correct]
        
        #print(party)
        #print(correct_class)
        return party, correct_class

def getEntropy(party, correct_class):
    correct_num = len(correct_class)
    if correct_num == 0 or correct_num == len(party):
        return 0
    p = correct_num / len(party)
    HS = - p * math.log(p, 2) - (1 - p) * math.log(1 - p, 2)
    return HS

def getErrorRate(party, correct_class):
   
    correct_num = len(correct_class)
    error_rate = correct_num/len(party)
    return error_rate if error_rate < 0.5 else 1 - error_rate
    	

with open(outfile,"w") as fr:   
    party, correct_class = read_parse(train_file)
    entropy = getEntropy(party,correct_class)
    error_rate = getErrorRate(party, correct_class)
    fr.write("entropy: " + str(entropy) + "\n" + "error: " + str(error_rate))



 #8改
#b把name放到inspection
#you can assume one - one