# Report
The code here use Q-learning (SARSAMAX) to solve the Banana environement
## DQN Agent
The state-action value function (Q) is represented by neural network (DQN) with 3 fully connected layers. The model weight is saved in `checkpoint.pth`

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
- Batch_size: 64
- Hidden_dim: 64
## Results
The environment is solved in 453 episodes.

![Training score](https://github.com/ccakarolotw/Udacity_Deep_RL_Navigation_Project/blob/main/train_score.png)

## Future ideas
- The learning rate might be improved by tweaking the hyperparameters and neural network structures. 
- Other form of SARSA (ie. SARSA(0) and Expected SARSA) can also be tested to see if they give better results.
- More improvements to Q-learning like double DQN, Prioritized experience replay, and dueling DQN are also woth testing. 


