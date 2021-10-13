import gym
import numpy as np
from gym import wrappers
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
import os
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, alpha=0.0003, gamma=0.92, n_actions=2, n_outputs=1):
        self.gamma = gamma
        self.n_actions = n_actions
        self.action = None
        self.log_probs = None
        self.n_outputs = n_outputs
        self.actor_critic = ActorCriticNetwork(n_actions=n_actions)
        self.actor_critic.compile(optimizer=Adam(learning_rate=alpha))

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        v,pi = self.actor_critic(state)
        mu, sigma = pi[0]
        sigma = tf.math.exp(sigma)

        action_probabilities = tfp.distributions.Normal(mu, sigma)
        action = action_probabilities.sample(sample_shape=(4,))
        log_prob = action_probabilities.log_prob(action)
        action = tf.math.tanh(action)
        self.action=action
        return action.numpy()

    def save_models(self):
        self.actor_critic.save_weights(self.actor_critic.checkpoint_file)
        print("Model Saved")

    def load_models(self):
        self.actor_critic.load_weights(self.actor_critic.checkpoint_file)
        print("Loaded Models")

    def learn(self, state, reward, state_, done):
        state = tf.convert_to_tensor([state],dtype=tf.float32)
        state_ = tf.convert_to_tensor([state_],dtype=tf.float32)
        reward = tf.convert_to_tensor(reward,dtype=tf.float32)
        with tf.GradientTape() as tape:
            state_value, probs = self.actor_critic(state)
            state_value_, _ = self.actor_critic(state_)
            state_value = tf.squeeze(state_value)
            state_value_ = tf.squeeze(state_value_)
            action_probs = tfp.distributions.Categorical(probs=probs)
            log_prob = action_probs.log_prob(self.action)
            delta = reward + self.gamma*state_value_*(1-int(done)) - state_value
            actor_loss = -log_prob*delta
            critic_loss = delta**2
            total_loss = actor_loss + critic_loss
        gradient = tape.gradient(total_loss, self.actor_critic.trainable_variables)
        self.actor_critic.optimizer.apply_gradients(zip(
            gradient, self.actor_critic.trainable_variables))


class ActorCriticNetwork(keras.Model):
    def __init__(self, n_actions=2, fc1_dims=1024, fc2_dims=512, name='ActorCritic', chkpt_dir='tmp/ActorCritic'):
        super(ActorCriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.v = Dense(1, activation='relu')
        self.pi = Dense(self.n_actions, activation='relu')

    def call(self, state):
        value = self.fc1(state)
        value = self.fc2(value)
        v = self.v(value)
        pi = self.pi(value)
        return v, pi

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

env = gym.make('BipedalWalker-v3')
agent = Agent(alpha=1e-5)
n_games = 50
filename = 'BipedalWalker_Graph.png'
figure_file = 'plots/' + filename
best_score = env.reward_range[0]
score_history = []
load_checkpoint = False
if load_checkpoint:
    agent.load_models()
for i in range(n_games):
    observation = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        score += reward
        if not load_checkpoint:
            agent.learn(observation, reward, observation_, done)
        observation = observation_
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])
    if avg_score > best_score:
        best_score = avg_score
        if not load_checkpoint:
            agent.save_models()
    print("Episode", i, "with score of",score, "and average score of", avg_score)
if not load_checkpoint:
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)
