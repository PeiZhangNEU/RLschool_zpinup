import gym
import metagym.quadrupedal
import numpy as np
import sys
sys.path.append('/home/zp/deeplearning/RL_School')

from alogos.sac.sac import sac
from alogos.sac.sac_train import train
import alogos.sac.core as core

env = gym.make('quadrupedal-v0',render=0,task="terrain")

# 设置超参数
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='unitree_a1')
parser.add_argument('--hid', type=int, default=256)
parser.add_argument('--l', type=int, default=2)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--seed', '-s', type=int, default=0)
parser.add_argument('--epochs', type=int, default=250)  # 一共训练了3e6次
parser.add_argument('--exp_name', type=str, default='sac_unitree_a1')
args = parser.parse_args()

# 执行训练过程
train(lambda : env, actor_critic=core.MLPActorCritic,
    ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
    seed=args.seed, epochs=args.epochs, expname=args.exp_name, use_gpu=True)
# o = env.reset()  # 只有在reset的时候，o才是一个字典！
# for i in range(100):
#     o, r, d, _ = env.step(env.action_space.sample())  # 在运行过程中，o不是字典
