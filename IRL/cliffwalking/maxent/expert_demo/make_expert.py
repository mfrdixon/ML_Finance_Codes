import gym
import gym_fcw
import readchar
import numpy as np


#env = gym.make('CliffWalking-v0')
env = gym.make('fcw-v0')

trajectories = []
episode_step = 0
START = [0,0]

for episode in range(20): # n_trajectories : 20
    trajectory = []
    step = 0
    env.reset()
    print(env.action_space)
    #print(env.step(0))
    print("episode_step", episode_step)
    state = START 
    print(state)
    done = False
    while done==False: 
        env.render()
        print("step", step)
        key = readchar.readkey()
        action = int(key) 
        state, reward, done = env.step(state, action)
        print(state)
        trajectory.append((env.s, action))
        step += 1

    trajectory_numpy = np.array(trajectory, float)
    print("trajectory_numpy.shape", trajectory_numpy.shape)
    episode_step += 1
    trajectories.append(trajectory)

np_trajectories = np.array(trajectories, dtype='int')
print("np_trajectories.shape", np_trajectories.shape)

np.save("expert_trajectories", arr=np_trajectories)
