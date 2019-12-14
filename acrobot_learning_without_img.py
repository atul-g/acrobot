# IMPORTS
import gym
import numpy as np
from collections import deque
import random
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential


# CONSTANTS
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)

MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 5_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False

# For stats
ep_rewards = [-200]



class DQNAgent:
    def __init__(self):

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0
    
    def create_model(self):
        model=tf.keras.Sequential([
                               tf.keras.layers.Dense(24, input_shape=(6,), activation='relu'),
                               tf.keras.layers.Dense(24, activation='relu'),
                               tf.keras.layers.Dense(3, activation="linear")
        ])
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episodes
    def train(self, terminal_state, step):
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
            
        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q
    
            # And append to our training data
            X.append(current_state)
            y.append(current_qs)
            
        # Fit on all samples as one batch only on terminal state
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1
            
        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
            
        # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        #observation shape is a (6,) vector which needs to be converted to a (1,6) array in order
      #to be passed into the predict function. We then use [0] index because the O/P is an array of [[_,_,_]].
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]



class Env:
    RETURN_IMAGES = True
    OBSERVATION_SPACE_VALUES = (6)  # 4
    ACTION_SPACE_SIZE = 3
    
    def __init__(self):
        self.env=gym.make('Acrobot-v1')
    
    def reset(self):
        self.episode_step = 0
        observation=self.env.reset()
        if self.RETURN_IMAGES:
            self.env.render()
        return observation

    def step(self, action):
        self.episode_step += 1
        
        new_observation, reward, done, _ = self.env.step(action)
        if self.RETURN_IMAGES:
            self.env.render()

        return new_observation, reward, done

agent = DQNAgent()
env=Env()
# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1
    
    # this state is only when it is in reset. rest of the time we will return image array
    current_state = env.reset()
    
    # Reset flag and start iterating until episode ends
    done = False
    
    while not done:

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, 3)

        new_state, reward, done = env.step(action)
        episode_reward+=reward
        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)
        
        current_state = new_state
        step += 1
        
    # Append episode reward to a list
    ep_rewards.append(episode_reward)
    
    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
        
        
        
agent.model.save('main_model.h5') 
agent.target_model.save('target_model.h5')
        
        
        
        
        
        
        



