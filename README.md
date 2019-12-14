# DQN for OpenAI Acrobot
This repository contains the code for creating a Deep Q-Network which helps the OpenAI gym's classic [Acrobot-v1](https://gym.openai.com/envs/Acrobot-v1/) to learn to cross a line by swinging back or forth from a pole.

img here


The code used is mainly derived from [here](https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/). A great source for learning about Deep Q-Networks and how to code them. I made a few changes to the code given in the link by using a classic gym environment and also made my own neural network model. Instead of RGB images of the frames of the game, I used direct observations values received from the environment while running each step. These tweaks were done to make the learning computationally less-expensive and to reduce the training time while also getting good results.

After training for 5000 episodes with a small dense neural network it was able to achieve an 100-episode average result of around -83. (Acrobot-v1 does not have a specified reward threshold at which it's considered solved.) 

The number of episodes and the dense neural network architecture can be changed to get better results.

At episode:
img

