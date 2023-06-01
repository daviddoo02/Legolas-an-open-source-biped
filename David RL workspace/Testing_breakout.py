import os
import gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy    
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold     

env = make_atari_env('Breakout-v0', n_envs=1, seed=0) 
env = VecFrameStack(env, n_stack=4)

A2C_path = os.path.join('Training', 'Saved Models', 'breakout_2M')

model = A2C.load(A2C_path, env)

eval = evaluate_policy(model, env, n_eval_episodes=50, render=True)
print(eval)