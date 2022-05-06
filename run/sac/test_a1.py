import gym
import metagym.quadrupedal
import numpy as np
import sys
sys.path.append('/home/zp/deeplearning/RL_School')
from alogos.sac.sac import sac
from alogos.sac.sac_train import train
import alogos.sac.core as core
from alogos.sac.sac_test import test



test('data/sac_unitree_a1/sac_249_ac.pt','quadrupedal-v0', "terrain")