
# coding: utf-8

# In[1]:


import gym
import gym_fcw
import numpy as np
import random

from maxent import *
import matplotlib
import matplotlib.pyplot as plt


# In[2]:


env = gym.make('fcw-v0')


# In[3]:


raw_demo = np.load(file="expert_demo/expert_trajectories.npy")



n_states = env.observation_space.n # position - 20, velocity - 20
n_actions = env.nA
one_feature = env.observation_space.n # number of state per one feature
q_table = np.zeros((n_states, n_actions)) # (400, 3)
feature_matrix = np.eye((n_states)) # (400, 400)

gamma = 0.99
q_learning_rate = 0.03
theta_learning_rate = 0.05

np.random.seed(1)
print(n_states)

# In[7]:


def idx_demo(env, one_feature):
    
    raw_demo = np.load(file="expert_demo/expert_trajectories.npy")
    demonstrations = np.zeros((len(raw_demo), len(raw_demo[0])))

    for x in range(len(raw_demo)):
        for y in range(len(raw_demo[0])):
            state_idx = raw_demo[x][y][0]

            demonstrations[x][y] = state_idx
            
    return demonstrations

def idx_state(env, state):
    
    state_idx = (4-1-state[0])*12 + state[1]
    return state_idx
# In[34]:


def update_q_table(state, action, reward, next_state):
    
    q_1 = q_table[state][action]
    q_2 = reward + gamma * max(q_table[next_state])
    q_table[state][action] += q_learning_rate * (q_2 - q_1)


# In[36]:


demonstrations = idx_demo(env, one_feature)

expert = expert_feature_expectations(feature_matrix, demonstrations)

learner_feature_expectations = np.zeros(n_states)

theta = -(np.random.uniform(size=(n_states,)))

episodes, scores = [], []

for episode in range(50001):
    state = [0,0] # env.reset()
    score = 0
    
    if episode % 1000 ==0:
        print("episode " + str(episode))
    if (episode != 0 and episode == 10000) or (episode > 10000 and episode % 1000 == 0):
        learner = learner_feature_expectations / episode
        maxent_irl(expert, learner, theta, theta_learning_rate)

    while True:
        state_idx = idx_state(env, state)
        
        action = np.random.choice(np.where(q_table[state_idx] == q_table[state_idx].max())[0]) 
    
        #action = np.argmax(q_table[state_idx])
        next_state, reward, done = env.step(state, action)
        
        #print(done, next_state, reward, action, state)
        irl_reward = get_reward(feature_matrix, theta, n_states, state_idx)
        next_state_idx = idx_state(env, next_state)
        update_q_table(state_idx, action, irl_reward, next_state_idx)

        learner_feature_expectations += feature_matrix[int(state_idx)]

        score += reward
        state = next_state

        if done:
            scores.append(score)
            episodes.append(episode)
            break

    if episode % 1000 == 0:
        score_avg = np.mean(scores)
        print('{} episode score is {:.2f}'.format(episode, score_avg))
        print('state, action 0, action 1, action 2')
        for i in range(len(q_table)):
            sum_=np.sum(q_table[i])
            if (sum_ !=0):
                print(i, q_table[i]/sum_)
            else:
                print(i, q_table[i])

plt.plot(episodes, scores)
plt.savefig('scores.png')
np.save("./results/maxent_q_table", arr=q_table)

