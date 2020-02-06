import sys
import environment as env
from environment import MountainCar as mc
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
global mode, weight_out, returns_out, episodes, max_iterations, epsilon, gamma, learning_rate

mode = sys.argv[1]#raw, tile
weight_out = sys.argv[2]
returns_out = sys.argv[3]
episodes = sys.argv[4]
max_iterations = sys.argv[5]
epsilon = sys.argv[6]
gamma = sys.argv[7]
learning_rate = sys.argv[8]

class LinearQNetwork(object):
    def __init__(self,state,gamma,learning_rate,mode):
        global weight, bias
        self.lr = learning_rate
        self.mode = mode              
        self.action = [0,1,2]
        self.dr = gamma
        self.lr = learning_rate
   
    def evaluate(self,state):
        global weight, bias,s,next_s
        q_value_array = []
        product = 0.0
        
        for action in self.action:
            product = np.dot(state,weight[:,action])
            q_value_array.append(product)
            
        for i in range(len(q_value_array)):
            q_value_array[i] += bias
        return q_value_array

    def update(self, q_value_array,  state, action, r, q_value_next):
        global weight, bias,s
        weight[:,action] -= self.lr * ( q_value_array[action] - (r + self.dr *  max(q_value_next))) * s
        bias -= self.lr * ( q_value_array[action] - (r + self.dr * max(q_value_next) ))  * 1
      
        
        return weight
    
class Agent(object):
    def __init__(self,epsilon):
        self.action = [0,1,2]
        self.epsilon = epsilon
     
    def policy(self, q_value_array):
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(self.action)
        else:
            action = np.argmax(q_value_array)  # Only the first occurrence is returned.
        return action
            
def main():
    global score_list,mode, weight_out, returns_out, episodes, max_iterations, epsilon, gamma, learning_rate,s,next_s,weight,bias
  
    episodes = int(episodes)
    max_iterations = int(max_iterations)
    epsilon = float(epsilon)
    gamma = float(gamma)
    learning_rate = float(learning_rate)
    agent = Agent(epsilon)
    bias = 0
    score_list = []
    mcr = mc(mode) 
    if mode == "raw":
        weight = np.zeros((2,3))
    else:
        weight = np.zeros((2048,3))
             
    for i in range(episodes): #one episode is a series of s -> a -> s' -> a' -> ... 
#        mcr = mc(mode) 
        state = mcr.reset() #{0:1,1:1,2:1,200,1} 
        q_network = LinearQNetwork(state,gamma,learning_rate,mode)
        if mode == "raw":
            s = np.zeros(2)
            for key in state:
                s[key] = state[key] #S [-0.553412, 0]
        if mode == "tile":
            s = np.zeros(2048)
            for key in state:
                s[key] = 1 #S [1,0,1,1,1]
        score = 0
        for i in range(max_iterations):
            q_value_array = q_network.evaluate(s) #[0,0,0] 这个state下所有action的q value
         
            action = agent.policy(q_value_array) #0,1,2 #Get action from the state #0
           
            next_state, reward, done = mcr.step(action) #if you reach the goal or not
            score += reward
            if mode == "raw":
                next_s = np.zeros(2)
                for key in next_state:
                    next_s[key] = next_state[key] #[1,0,1,1,1]
                
            if mode == "tile":
                next_s = np.zeros(2048)
                for key in next_state:
                    next_s[key] = 1  #{1:1,2:1}
            
            q_value_array_next = q_network.evaluate(next_s) 
            
            w = q_network.update(q_value_array, state, action, reward, q_value_array_next)
            s = next_s
           
            if done:
#                state = mcr.reset() #{0:1,1:1,2:1,200,1}
                break
        score_list.append(score)
    with open (returns_out,"w") as f:
        for i in score_list:
            f.write(str(i) + "\n")

    weight_list = [col for row in w for col in row]
    weight_list.insert(0,bias)

    with open (weight_out,"w") as f2:
        for i in weight_list:
            f2.write(str(i) + "\n")

    x_axis = [i for i in range(400)]
    a = pd.Series(score_list)
    b = a.rolling(25).mean()
    c = b.to_numpy()
    plt.plot(x_axis,score_list)
    plt.plot(x_axis,c)
    plt.ylabel('number of episodes')
    plt.ylabel('sum of rewards in an episode')
    plt.title("Tile Features")
    plt.show()

if __name__ == "__main__":
    main()
#cmd for printing plot
#python q_learning.py raw weight.out returns.out 2000 200 0.05 0.999 0.001
#python q_learning.py tile weight.out returns.out 400 200 0.05 0.99 0.00005
    
