# DQN for OpenAI Acrobot
Creating a Deep Q-Network which helps the OpenAI gym's classic [Acrobot-v1](https://gym.openai.com/envs/Acrobot-v1/) to learn to cross a line by swinging back and forth on a pole.

![imghere](https://raw.githubusercontent.com/atul-g/acrobot/master/sample_acrobot%20.png)

## Code
The code used is mainly derived from [here](https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/). A great source for learning about Deep Q-Networks and how to code them. I made a few changes to the code given in the link by using a classic gym environment and also made my own neural network model. Instead of RGB images of the frames of the game, I used direct observations values received from the environment while running each step. These tweaks were done to make the learning computationally less-expensive and to reduce the training time while also getting good results.

After training for 5000 episodes with a small dense neural network, it was able to achieve a 100-episode average result of around -83. (Acrobot-v1 does not have a specified reward threshold at which it's considered solved.) 

The number of episodes and the dense neural network architecture can be changed to get better results. Open `acrobot_learning_without_img.py` for the code.

## Preview
#### At episode 940:

![ep_940](https://raw.githubusercontent.com/atul-g/acrobot/master/episode_940.gif)


#### At episode 2500:
![ep_2500](https://raw.githubusercontent.com/atul-g/acrobot/master/episode_2500.gif)

#### At episode 5000:
![ep_5000](https://raw.githubusercontent.com/atul-g/acrobot/master/episode_5000.gif)


## Using the model:
The model has been saved as `main_model.h5`. It can be used elsewhere using Keras's `model.load_model` function and used to play
the games again. The corresponding file doing this is `playing_acrobot.py`.
