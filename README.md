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

## DQN Agent
The state-action value function (Q) is represented by neural network (DQN) with 3 fully connected layers

```
x = nn.Linear(state_dim,hidden_dim)(state)
x = F.relu(x)
x = nn.Linear(hidden_dim, hidden_dim)(x)
x = F.relu(x)
x = nn.Linear(hidden_dim, action_dim)(x)
```

The agent uses the output of the DQN and choose action based on epsilon-greedy policy.

The DQN is trained based on Q learning. 

Q(state, action) = reward + GAMMA* max(Q'(next_state,)) 

Q' is the target Q network.

### Hyperparameters
- epsilon: starts from 1 and is multiplied by 0.99 after each action taken and then fixed at 0.01.
- GAMMA: 0.99
- optimizer: Adam with learning rate 5E-4
- Q' is update with soft update after each update of Q (Q'_ parameter = (1-tau)* Q'_ parameter + tau* Q_ parameter)
- tau: 1E-3
## Results
The environment is solved in 453 episodes.

![Training score](https://github.com/ccakarolotw/Udacity_Deep_RL_Navigation_Project/blob/main/train_score.png)

