import gym
import readchar
import numpy as np

# # MACROS
##Push_Left = 0
#No_Push = 1
#Push_Right = 2
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

# Key mapping
arrow_keys = {
    '\x1b[A': UP,
    '\x1b[C': RIGHT,
    '\x1b[B': DOWN,
    '\x1b[D': LEFT}

#env = gym.make('MountainCar-v0')
env = gym.make('CliffWalking-v0')

trajectories = []
episode_step = 0

for episode in range(5): # n_trajectories : 20
    trajectory = []
    step = 0

    env.reset()
    print(env.action_space)
    print(env.step(0))
    print("episode_step", episode_step)

    while True: 
        env.render()
        print("step", step)

        key = readchar.readkey()
        print(key)
        #if key not in arrow_keys.keys():
        #    break

        action = int(key) #arrow_keys[key]
        
        state, reward, done, _ = env.step(action)

        if state == 47 or step>=12: #env.goal_position: # and step > 129: # trajectory_length : 130
            break
        print(state)
        trajectory.append((state, action))
        step += 1

    trajectory_numpy = np.array(trajectory, float)
    print("trajectory_numpy.shape", trajectory_numpy.shape)
    episode_step += 1
    trajectories.append(trajectory)

np_trajectories = np.array(trajectories, dtype='int')
print("np_trajectories.shape", np_trajectories.shape)

np.save("expert_trajectories", arr=np_trajectories)
