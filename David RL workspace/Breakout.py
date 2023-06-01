"""
There are five steps to do RL:
    1. import dependencies
    2. test environment
    3. Vectorise Environment and Train model
    4. Save and reload model
    5. Evaluate and test

"""
# 1. Import dependencies:


import gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold   
import os


# 2. test environment

"""
# Create environment

environment_name = "Breakout-v0"
env = gym.make(environment_name)

episodes = 5                    
for episode in range(1, episodes+1):                # looping through 5 episodes
    obs = env.reset()                               # Make observation from the environment
    done = False
    score = 0

    while not done:
        env.render()                                    # view the environment
        action = env.action_space.sample()              # Generate random action
        obs, reward, done, info = env.step(action)      # Apply random action to environment through function step()
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()

"""

# Setting up 4 environments to train at once
env = make_atari_env('Breakout-v0', n_envs=4, seed=0)       # make_atari_env - create wrapped Atari env
env = VecFrameStack(env, n_stack=4)                         # VecFrameStack - stack environments together

# 3. Train a model:

log_path = os.path.join("Training", "Logs")             # Define where to save training informations - Note: Make the directories first

model = A2C('CnnPolicy', env, verbose=1, tensorboard_log=log_path)      # Define our model: MlpPolicy = Multi Layered Perceptron Policy

model.learn(total_timesteps=2000000)                      # Train our model

save_path = os.path.join("Training", 'Saved Models', 'breakout_2M')

# ----------------- Callbacks ----------------------------- #
"""

# Call back defining training goal

stop_callback = StopTrainingOnRewardThreshold(reward_threshold=500, verbose=1)

# Callback function to evaluate the training and stop training when goal is met
eval_callback = EvalCallback(env, callback_on_new_best=stop_callback, eval_freq=10000, best_model_save_path=save_path, verbose=1)

# ----------------- Important ----------------------------- #

"""

model.save(save_path)                                    # Save our model