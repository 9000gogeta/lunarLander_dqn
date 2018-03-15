#Algorithm converges although it requires furher tuning to consistently place around 250

from __future__ import print_function
import gym
import tensorflow as tf
import numpy as np
from itertools import count
from replay_memory import ReplayMemory, Transition
import env_wrappers
import random
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--eval', action="store_true", default=False, help='Run in eval mode')
parser.add_argument('--seed', type=int, default=1, help='Random seed')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

class DQN(object):

    #A class to implement the Deep Q Network algorithm
    def __init__(self, env):

        self.env = env
        self.sess = tf.Session()

        # A few starter hyperparameters
        self.batch_size = 128
        self.gamma = 0.99
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 1000
        self.decay_rate = (self.eps_start-self.eps_end)/self.eps_decay
        self.clone_steps = 5000
        self.replay_memory = ReplayMemory(100000)
        self.min_replay_size = 10000
        self.learning_rate = 0.001
        self.target_update=100

        # Define placeholder for input
        self.observation_input = tf.placeholder(tf.float32,shape = [None,8])
        self.build_model(self.observation_input)

        self.ind = tf.stack([tf.range(tf.shape(self.action)[0], dtype=tf.int32), self.action], 1)
        self.qvals = tf.gather_nd(params=self.q, indices=self.ind) 

        #Compute loss of label(qvalues) and prediction(qvals)
        #Optimize with 0.001 learning rate
        self.loss = tf.losses.huber_loss(self.qvalues, self.qvals)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.num_episodes = 0
        self.num_steps = 0

        self.saver = tf.train.Saver(tf.trainable_variables())
        self.sess.run(tf.global_variables_initializer())

    def build_model(self, observation_input, scope='train'):

        #Define the placeholders and output q-values
        #Used 1 hidden layer(fully connected/dense) for both the normal and target networks
        #Output Q-values according to the Q-learning rule

        self.action = tf.placeholder(tf.int32,shape = [None,])
        self.reward = tf.placeholder(tf.float32,shape = [None,])
        self.new_input = tf.placeholder(tf.float32,shape = [None,8])

        with tf.variable_scope('q'):
            self.hidden_eval = tf.contrib.layers.fully_connected(self.observation_input, 500, tf.nn.relu)
            self.q = tf.contrib.layers.fully_connected(self.hidden_eval, 4)

        with tf.variable_scope('q_target'):
            self.hidden_target = tf.contrib.layers.fully_connected(self.new_input, 500, tf.nn.relu,trainable=False)
            self.q_target = tf.contrib.layers.fully_connected(self.hidden_target, 4,trainable=False)

        self.qvalues = self.reward+self.gamma*tf.reduce_max(self.q_target,1)
        


    def select_action(self, obs, evaluation_mode=False):

        #Explore if random.random < self.epsilon
        #Exploit otherwise
        if random.random() < self.epsilon and evaluation_mode==False:
            return self.env.action_space.sample()
        else: 
            q_values = self.sess.run(self.q,feed_dict={self.observation_input:obs.reshape(1,8)})
            action = np.argmax(q_values)
            return action

    def update(self):

        #Backup parameters to target network every 100 steps
        if self.num_steps % self.target_update == 0:
            training = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_target')
            network = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q')
            self.sess.run([tf.assign(t, n) for t, n in zip(training, network)])

        #Sample a batch from replay memory and feed into the neural network
        minibatch = self.replay_memory.sample(self.batch_size)
        obs = [minibatch[i].state for i in range(self.batch_size)]
        act = [minibatch[i].action for i in range(self.batch_size)]
        rew = [minibatch[i].reward for i in range(self.batch_size)]
        new_obs = [minibatch[i].next_state for i in range(self.batch_size)]
        self.sess.run(self.optimize,{self.observation_input:obs,self.action:act,self.reward:rew,self.new_input:new_obs})
    
    def train(self):

        #Add transitions to replayMemory
        #Update network when minimum number of samples have been added to the replay memory
        done = False
        obs = env.reset()
        self.epsilon  = self.eps_start
        self.num_steps = 0
        while not done:
            action = self.select_action(obs, evaluation_mode=False)
            next_obs, reward, done, info = env.step(action)
            self.replay_memory.push(obs,action,next_obs,reward,done)
            if len(self.replay_memory) > self.min_replay_size:
                self.update()
            self.epsilon -= self.decay_rate
            self.num_steps += 1
            obs = next_obs
        self.num_episodes += 1

    def eval(self, save_snapshot=True):

        #Run an evaluation episode, this will call
        total_reward = 0.0
        done = False
        obs = env.reset()
        while not done:
            env.render()
            action = self.select_action(obs, evaluation_mode=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        print ("Evaluation episode: ",total_reward)
        if save_snapshot:
            print ("Saving state with Saver")
            self.saver.save(self.sess, 'models/dqn-model', global_step=self.num_episodes)

def train(dqn):
    for i in count(1):
        dqn.train()
        # every 10 episodes run an evaluation episode
        if i % 10 == 0:
            dqn.eval()

def eval(dqn):

    #Load the latest model and run a test episode
    ckpt_file = os.path.join(os.path.dirname(__file__), 'models/checkpoint')
    with open(ckpt_file, 'r') as f:
        first_line = f.readline()
        model_name = first_line.split()[-1].strip("\"")
    dqn.saver.restore(dqn.sess, os.path.join(os.path.dirname(__file__), 'models/'+model_name))
    dqn.eval(save_snapshot=False)


if __name__ == '__main__':

    env = gym.make('LunarLander-v2')
    env.seed(args.seed)

    dqn = DQN(env)
    if args.eval:
        eval(dqn)
    else:
        train(dqn)
