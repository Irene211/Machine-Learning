# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 11:11:37 2019

@author: Yelena
"""
import sys
import math
#train_file = sys.argv[1]
#test_file = sys.argv[2]
#max_depth = sys.argv[3]
#train_label = sys.argv[4]
#test_label = sys.argv[5]
#metrics = sys.argv[6]
global correct, res
correct = ["democrat","y","A","before1950","yes","morethan3min","fast","expensive","high","Two","large"]
res = []
def read_parse(file):
    with open(file) as f:
        content = f.readlines()
        content = [x.strip("\r\n") for x in content]
        content = [x.strip("\n") for x in content]
        content = [x.strip(" ") for x in content]
        
        total = [i.split("\t") for i in content]

        attr_name = content[0].split("\t")
        array = [[party for party in line.split("\t")] for line in content[1:]]
        total_val_array = [[1 if i in correct else 0 for i in line] for line in array]

        return attr_name, total_val_array, total

def res_entropy(total_val_array):
    res_list = [line[-1] for line in total_val_array]
    count = 0

    for i in res_list:
        count += i
        
    if count == 0 or count == len(res_list): 
        return 0
    p = count/len(res_list)
    HS = - p * math.log(p, 2) - (1 - p) * math.log(1 - p, 2)
    return HS
    
def attr_entropy(total_val_array,cur_col):

    res_list = [line[-1] for line in total_val_array]
   
    attr_list = [line[cur_col] for line in total_val_array]
    if len(res_list) != len(attr_list):
        raise Exception("Lenth of column is not equal or data lost")
        
    zero_one, zero_zero, one_one, one_zero = 0, 0, 0, 0
    for row in range(len(res_list)):
        if attr_list[row] == 0 and res_list[row] == 1:
            zero_one += 1
        elif attr_list[row] == 0 and res_list[row] == 0:
            zero_zero += 1
        elif attr_list[row] == 1 and res_list[row] == 1:
            one_one += 1
        elif attr_list[row] == 1 and res_list[row] == 0:
            one_zero += 1
  
    attr_zero = zero_one + zero_zero
    attr_one = one_zero + one_one

    if attr_zero == 0:
        P_ZERO_ONE = 0
    else:
        P_ZERO_ONE = zero_one/attr_zero
    if attr_one == 0:
        P_ONE_ONE = 0
    else:
        P_ONE_ONE = one_one/attr_one
 
    if zero_one == 0 or zero_one == attr_zero:
        HS_ZERO = 0
    else:
         HS_ZERO = - P_ZERO_ONE * math.log(P_ZERO_ONE, 2) - (1 - P_ZERO_ONE) * math.log(1 - P_ZERO_ONE, 2)
    
    if one_one == 0 or one_one == attr_one:
        HS_ONE = 0
    else:
        HS_ONE = - P_ONE_ONE * math.log(P_ONE_ONE, 2) - (1 - P_ONE_ONE) * math.log(1 - P_ONE_ONE, 2)
    
    P_ZERO = attr_zero/len(res_list)

    HS = P_ZERO * HS_ZERO + (1 - P_ZERO) * HS_ONE 
    
    return HS

def info_gain(total_val_array,attr_index):
    result_entropy = 0
    result_entropy = res_entropy(total_val_array)
    Info_gain = []
    if len(attr_index) == 1:
        return len(total_val_array) - 1, 0
    else:
        for i in attr_index[:-1]:
            attribute_entropy = 0
            attribute_entropy = attr_entropy(total_val_array, i)
     
            col_info_gain = result_entropy - attribute_entropy
            
            Info_gain.append(col_info_gain)
             
        max_info_gain = max(Info_gain)
        col = Info_gain.index(max_info_gain)
        col = attr_index[col]
    
    return col,  max_info_gain

class Node:
    def __init__(self, label=None, attr=None, left=None, right=None, one=0, zero = 0):
        self.current_col= -1
        self.label = label
        self.attr = attr
        self.left= left
        self.right= right
        self.one = one
        self.zero = zero
        
def print_pre_order(root,depth,w, max_dep,content):
    first_col =  [i[0] for i in content][1:]
    last_col = [i[-1] for i in content][1:]
    one = [i for i in first_col if i in correct][0]
    zero = [i for i in first_col if i not in correct][0]
    
    res_one = [i for i in last_col if i in correct][0]
    res_zero = [i for i in last_col if i not in correct][0]

    if depth > max_dep:
        return
    if isinstance(root,Node):
        tree = ""
        if depth == 0:
            tree += "[" + str(root.one) + " " + res_one + " /" + str(root.zero) + " " + res_zero + "]"
        
        if depth > 0:
            if depth > 1:
                if depth == 2:
                    tree += "| "
                if depth == 3:
                    tree += "| " + "| "
            if w == "this_is_left":
                tree += "| "+str(root.attr) + " = " + one + ": [" + str(root.one) + " " + res_one + " /" + str(root.zero) +  " " + res_zero + "]"
            else:
                tree += "| " + str(root.attr) + " = " + zero + ": [" + str(root.one)+ " " + res_one + " /" + str(root.zero) + " " + res_zero + "]"
            
        print(tree)
        print_pre_order(root.left,depth+1, "this_is_left", max_dep,content)
        print_pre_order(root.right,depth+1,"this_is_right", max_dep,content)

def main_body(total_val_array,attr_index, depth, attr, first_row, max_dep):   
    if depth > max_dep:
        return
    last_col = [i[-1] for i in total_val_array]
    is_one = 0  
    for i in last_col:
        if i == 1:
            is_one += 1    
    is_zero = len(last_col) - is_one  
    if is_one == 0 or is_one == len(last_col):

        is_attr = attr
        if is_one == 0:
            is_label = 0
        elif is_one == len(last_col):
            is_label = 1
        root = Node(label=is_label, attr = is_attr, one=is_one, zero=is_zero)
        return root
      
    is_label = 0 if is_zero > is_one else 1
    
    is_attr = attr
    root = Node(label=is_label, attr = is_attr, one=is_one, zero=is_zero)
    
    if len(attr_index) != 1:

        col, Info_gain = info_gain(total_val_array, attr_index)  
        root.current_col = col
        
        total_left = []
        total_right = []
   
        for i in total_val_array:
            if i[col] == 1:
               
                total_left.append(i)
            else:
              
                total_right.append(i)

        if len(total_left) == 0:
            root.label = 0 if is_zero > is_one else 1
            return root
        if len(total_right) == 0:
            root.label = 0 if is_zero > is_one else 1
            return root
        
        next_attr_index = [index for index in attr_index if index != col]
        
        if Info_gain > 0:
            attr_name = first_row
            current_attr_name = attr_name[col]
            root.left = main_body(total_left, next_attr_index, depth+1, attr = current_attr_name, first_row = attr_name,max_dep=max_dep) #1,Y
            root.right = main_body(total_right, next_attr_index, depth+1, attr = current_attr_name,first_row = attr_name,max_dep=max_dep)
    
    return root

def track_label(root,input_line):
    if isinstance(root,Node):
        if input_line[root.current_col] == 1:
            res.append(root.label)
            print("resleft",res)
            track_label(root.left, input_line)
        else:
            res.append(root.label)
            print("resright",res)
            track_label(root.right, input_line)

def output(train_file, test_file, max_depth, train_label, test_label, metrics):
    max_depth = int(max_depth)
    attr_name, total_val_array,total = read_parse(train_file)  
    attr_index = [] 
    for i in range(len(total_val_array[0])):
        attr_index.append(i)
        
    root = main_body(total_val_array, attr_index, 0, attr=None, first_row=attr_name, max_dep = max_depth)
    
    print_pre_order(root, 0, "this_is_left", max_dep = max_depth, content=total) 
    
    error_train = 0
    final = []
    for i in total_val_array:
        track_label(root, i)
        final.append(res[-1])
#        print("res:",res)
#        print("length of res:",len(res))
        if res[-1] != i[-1]:
            error_train += 1
    error_train = error_train/len(total_val_array)
    
    last_col = [i[-1] for i in total][1:]
    res_one = [i for i in last_col if i in correct][0]
    res_zero = [i for i in last_col if i not in correct][0]
    
    final=[res_one if x == 1 else res_zero for x in final]
    
    with open(train_label,"w") as fr:
        for line in final:     
            fr.write(line + "\n")

    test_attr_name, test_val_array,test_total = read_parse(test_file)  
    
    error_test = 0
    test_final = []
    for i in test_val_array:
        track_label(root, i)
        test_final.append(res[-1])
        if res[-1] != i[-1]:
            error_test += 1
    error_test = error_test/len(test_val_array)
    
    last_col = [i[-1] for i in test_total][1:]
    res_one = [i for i in last_col if i in correct][0]
    res_zero = [i for i in last_col if i not in correct][0]
    
    test_final=[res_one if x == 1 else res_zero for x in test_final]
    
    with open(test_label,"w") as fr:
        for line in test_final:     
            fr.write(line + "\n")
            
    with open(metrics,"w") as fr:
        error = "error(train): " + str(error_train) +"\n" +  "error(test): " + str(error_test)
        fr.write(error)
        
output("small_train.tsv","small_test.tsv", 3, "pol_2_train2.tsv" ,"pol_2_test2.tsv", "pol_2_metrics2.txt" )  

#output(train_file,test_file, max_depth, train_label , test_label,metrics )



