import sys
sys.path.append('/home/zp/deeplearning/RL_School')
# We show a simple example to start Quadrupedal here
import gym
import metagym.quadrupedal
import numpy as np
env = gym.make('quadrupedal-v0',render=1,task="stairstair",ETG=True,ETG_path="ESStair_origin.npz")
observation, info = env.reset()   # 之前observation不是单纯array的原因找到了，因为reset 的时候，这里是有两个输出的！
for i in range(100):
    action = np.random.uniform(-0.3,0.3,size=12)
    next_obs, reward, done, info = env.step(action)
