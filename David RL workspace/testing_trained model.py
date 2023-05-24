import os
import gym
from stable_baselines3 import PPO                                       # PPO is an algorithm included in stable baseline
from stable_baselines3.common.vec_env import DummyVecEnv                # DummyVecEnv is a wrapper
from stable_baselines3.common.evaluation import evaluate_policy    
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold     

environment_name = "CartPole-v0"
env = gym.make(environment_name)

save_path = os.path.join('Training', 'Saved Models')

# ----------------- Important ----------------------------- #

# Call back defining training goal
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)

# Callback function to evaluate the training and stop training when goal is met
eval_callback = EvalCallback(env, callback_on_new_best=stop_callback, eval_freq=10000, best_model_save_path=save_path, verbose=1)

# ----------------- Important ----------------------------- #

log_path = os.path.join("Training", "Logs")            

env = DummyVecEnv([lambda: env])                       
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path) 

model.learn(total_timesteps=20000, callback=eval_callback)