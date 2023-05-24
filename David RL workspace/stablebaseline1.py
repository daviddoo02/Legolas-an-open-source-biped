import os
import gym
from stable_baselines3 import PPO                                       # PPO is an algorithm included in stable baseline
from stable_baselines3.common.vec_env import DummyVecEnv                # DummyVecEnv is a wrapper
from stable_baselines3.common.evaluation import evaluate_policy         

# 1. Define training environment:

environment_name = "CartPole-v0"
env = gym.make(environment_name)

# -----------------------------------------------------------------------------------------------------------------------------------
# 2. Test out environment:

"""

episodes = 5                    
for episode in range(1, episodes+1):                # looping through 5 episodes
    state = env.reset()                             # Make observation from the environment
    done = False
    score = 0

    while not done:
        env.render()                                    # view the environment
        action = env.action_space.sample()              # Generate random action
        n_state, reward, done, info = env.step(action)  # Apply random action to environment through function step()
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()

"""

# Notes: Stable baseline only deal with "Model-Free RL"
#        Which types of algorithm you use is dependent on what type of space your environment is operating with
#        Which types of algorithm you used dictates your training metrics 
#        Training metrics: 
#               Evaluation metric: length of episode, numbers of reward

# -----------------------------------------------------------------------------------------------------------------------------------

# 3. Train a model:

log_path = os.path.join("Training", "Logs")             # Define where to save training informations - Note: Make the directories first

env = DummyVecEnv([lambda: env])                        # Create environment with the lambda function, then wrap the environment in a Dummy Vectorized Environment
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)      # Define our model: MlpPolicy = Multi Layered Perceptron Policy

# model.learn(total_timesteps=40000)                      # Train our model

# 4. Save trained model:

PPO_Path = os.path.join("Training", 'Saved Models', 'PPO_Model_Cartpole')

# model.save(PPO_Path)                                    # Save our model

# 5. Load model

model = PPO.load(PPO_Path, env=env)

# 6. Evaluation

evaluate_policy(model, env, n_eval_episodes=10, render=True)