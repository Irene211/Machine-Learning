# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 18:50:03 2019

@author: Yelena
"""
import sys
in_train = sys.argv[1]
in_valid = sys.argv[2]
in_test = sys.argv[3]
in_dict = sys.argv[4]
out_train = sys.argv[5]
out_valid = sys.argv[6]
out_test = sys.argv[7]
feature_flag = sys.argv[8]

def infile(file, output_file, file_dict, feature_flag):
    
    dict_txt = dict()
    with open(file_dict) as f_dict:
        txt = f_dict.readlines()
        txt = [x.strip("\n") for x in txt]
        for line in txt:
            line = line.split(" ")
            dict_txt[line[0]] = line[1]
    
    with open(file) as f:
        content = f.readlines()
        content = [x.strip("\n") for x in content]
        content = [x.replace("\t"," ") for x in content]
        content = [x.split(" ") for x in content]

    res = ""
    ref = dict()
    for paragraph in content:  
        res += paragraph[0] + "\t"
        for i in paragraph[1:]: #i is word
            
            if i in dict_txt:
              
                if dict_txt[i] in ref:
                    ref[dict_txt[i]] += 1
                else:
                    if int(feature_flag) == 1:
                     
                        res += dict_txt[i] + ":" + "1" + "\t"
                    ref[dict_txt[i]] = 1
                    
        if int(feature_flag) == 2:
            for i in ref:
                if ref[i] < 4:
                    res += str(i) + ":" + "1" + "\t"
        ref = dict()
        
        res += "\n"
        
    with open(output_file, "w") as fr:
        fr.write(res)

if __name__ == "__main__":
    for j in range(1,4):
#    infile("C:/Users/Yelena/Desktop/10601/601 - hw4/handout/smalldata/smalltrain_data.tsv", \
#           "C:/Users/Yelena/Desktop/10601/601 - hw4/handout/smalldata/formatted_train.tsv",\
#           "dict.txt", 1)
        infile(sys.argv[j], sys.argv[j+4], sys.argv[4], sys.argv[8])

#python feature.py smalltrain_data.tsv smallvalid_data.tsv smalltest_data.tsv dict.txt formatted_train.tsv formatted_valid.tsv formatted_test.tsv 1
#Desktop\10601\601 - hw4\handout\smalldata