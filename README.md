# Udacity_Deep_RL_Navigation_Project
This project uses deep Q learning to solve Udacity Deep Reinforcement Learning class's Unity Banana environment.The deep Q network agent is implemented based on Pytorch in `Navigation.ipynb`. The code is in PyTorch (v0.4) and Python 3. 

## The Unity Banana Environment
![Image of Unity Banana environment](https://github.com/ccakarolotw/Udacity_Deep_RL_Navigation_Project/blob/main/banana.gif)

The agent get +1 reward for getting a yellow banana and -1 for getting a blue banana. Each episode lasts 300 time stamps. The state space has 37 dimensions and the action space has 4 dimensions. The environement is considered solved when the agent receives an average reward (over 100 episodes) > 13.
  
## Dependencies
Installation of dependencies follow https://github.com/udacity/deep-reinforcement-learning#dependencies
1. Create (and activate) a new environment with Python 3.6.

`conda create --name drlnd python=3.6 
activate drlnd`

2.  Install pytorch 0.4.0

`conda install pytorch=0.4.0 -c pytorch`

3. Download Unity environment and place it in the same folder as the jupyter notebook `Navigation.ipynb`

## Results
