import gym
import metagym.quadrupedal
import numpy as np
import sys
sys.path.append('/home/zp/deeplearning/RL_School')
from alogos.ppo.ppo import ppo
from alogos.ppo.ppo_train import train
import alogos.ppo.core as core
from alogos.ppo.ppo_test import test



test('data/ppo_unitree_a1/ppo_110_ac.pt','quadrupedal-v0', "terrain")