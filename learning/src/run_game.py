#!/usr/bin/env python2

import gym
import os
import sys
import rospy
import rospkg
from q_learner import QLearner
import matplotlib.pyplot as plt
import shutil
import numpy as np

"""
Usage:
python run_game.py <env name> <file directory to save results> <optional: learned Q-value file>
This will train a tabular Q-learning agent on the provided environment (with given parameters) and save the results into the provided directory.
If a learned Q-value file is provided, you can watch the agent play the game with the learned Q-value function.
e.g. python run_game.py "TargetCatcher-v0" save_dir target_task_learnedQ/Q.csv
"""

def run(env, agent, render, save_dir, num_episodes):
    max_timesteps = 100
    save_freq = 100 # Q-values and plot will be updated at this frequency
    interval = 100 # Reward plot will only plot rewards at this frequency
    all_rewards = []
    mean_rewards = []
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    for i_episode in range(num_episodes):
        total_reward = 0
        state = env.reset()
        done = False
        for t in range(max_timesteps):
            if render:
                env.render()

            action = agent.select_action(state)
            next_state, reward, done, _info = env.step(action)

            agent.updateQ(state, action, reward, next_state)
            total_reward += reward
            state = next_state
            if done:
                break

        print('EPISODE: ', i_episode)
        print('TOTAL: ', total_reward)
        print('e: ', agent.e)
        print('\n')

        all_rewards.append(total_reward)
        # Keep track of mean reward over last few episodes (smoother learning curve)
        mean_rewards.append(float(np.mean(all_rewards[-interval:])))
        if i_episode % save_freq == 0:
            x = range(i_episode+1)[::interval]
            mean_rewards_y = mean_rewards[::interval]

            agent.saveQ(save_dir)
            agent.save_debug_info(save_dir)

            fig2 = plt.figure(1)
            ax2 = fig2.add_subplot(1,1,1)
            ax2.clear()
            ax2.plot(x, mean_rewards_y)
            plt.savefig(os.path.join(save_dir, "mean_rewards.png"))

if __name__ == "__main__":
    rospy.init_node('learning')
    env = gym.make('Learning-v0')

    # Replay policy
    # sourceQ_file = os.path.join(rospkg.RosPack().get_path('learning'), 'csv/sim', 'replay_policy', 'Q.csv') 
    sourceQ_file = None
    if sourceQ_file != None:
        save_dir = os.path.join(rospkg.RosPack().get_path('learning'), 'csv/sim', 'replay_policy')
    else:
        save_dir = os.path.join(rospkg.RosPack().get_path('learning'), 'csv/sim', 'learn_policy_v7') 

    # Additional parameters
    agent = QLearner(env) #, sourceQ_file=sourceQ_file) # include sourceQ_file if you want to replay a policy
    render = False
    num_episodes = 500000 # 200 episodes per minute

    run(env, agent, render, save_dir, num_episodes)

    rospy.spin()