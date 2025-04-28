import torch as T 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from model import QNetwork
import random
from collections import namedtuple, deque

#         Params
GAMMA=0.9
ALPHA=0.1
BATCH_SIZE=64 # minibatch size
BUFFER_SIZE = int(1e5)# replay buffer size
UPDATE_EVERY=4 # how often to update the network
TAU = 1e-3  

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

class DeepQNetwork():
    def __init__(self,state_size, action_size, seed):
        
        self.nstate=state_size
        self.naction=action_size
        self.seed=seed
        self.tstep=0

        self.Q=QNetwork(self.nstate, self.naction, seed)# Q predicted
        self.Qcap=QNetwork(self.nstate, self.naction, seed)#Q actual

        self.replaybuffer=Buffer()

        self.optimizer=optim.Adam(self.Q.parameters(),0.0005)


    def act(self,state,eps=1):
        #Function to chose the next action after the state is given
        r=random.random()
        if isinstance(state, tuple):
            state = state[0]
        if r<eps:
            return random.choice(range(self.naction))
        else:
            state = T.from_numpy(state).float().unsqueeze(0).to(device)
            self.Q.eval()#Sets the module in evaluation mode.
            with T.no_grad(): # used when we are not doing back propogation
                # to save memory
                action=self.Q.forward(state)
            self.Q.train()#Sets the module back to training mode.
            
            return np.argmax(action.cpu().numpy())
    def learn(self,experiences,GAMMA):
        # Function to learn the model
        
        #conversion of parameters from list comprehension to tensor
        state = T.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        action = T.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        reward = T.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_state = T.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        done = T.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).int().to(device)

        # state_shape torch.Size([64, 8])
        # action_shape torch.Size([64, 1])
        # reward_shape torch.Size([64, 1])
        # next_state_shape torch.Size([64, 8])
        # done_shape torch.Size([64, 1])
        

        y = reward.clone()  # Initialize y with rewards
        # Compute Q-learning target: Q(s, a) = r + γ max(Q(s', a'))
        with T.no_grad():  # No gradient tracking for Q_target
            next_Q_values = self.Qcap.forward(next_state)  # Get Q-values for next states
            max_next_Q = next_Q_values.max(dim=1)[0].unsqueeze(1)  # Get max Q-value for each next state
            y[~done] += GAMMA * max_next_Q[~done]  # Only update non-terminal states

        # Compute predicted Q-values
        # gather-->Picks the value along the given dimension from a index tensor
        # squeeze-->
        #           x = torch.tensor([[[5]], [[10]], [[15]]])  # Shape: (3, 1, 1)
        #           print(x.shape)  # Output: torch.Size([3, 1, 1]) 
        current_Q = self.Q.forward(state).gather(1, action).squeeze(1)
        y=y.squeeze(1)

        # Compute MSE loss
        loss = F.mse_loss(current_Q, y)

        self.optimizer.zero_grad()# resets gradient
        loss.backward()# calculates the gradient descent 
        self.optimizer.step() # updates the parameters of the model

            
    def step(self,state, action, reward, next_state, done):
        # Save experience in replay memory
        self.replaybuffer.add(state, action, reward, next_state, done)
        
        
        self.tstep=(self.tstep+1)%UPDATE_EVERY

        if self.tstep==0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.replaybuffer)>BATCH_SIZE:
                experiences=self.replaybuffer.sample()
                self.learn(experiences, GAMMA)
                for target_param, local_param in zip(self.Qcap.parameters(), self.Q.parameters()):
                    target_param.data.copy_(TAU*local_param.data + (1.0-TAU)*target_param.data)

class Buffer():
    def __init__(self):
        self.buffer=deque(maxlen=BUFFER_SIZE)
        self.experience=namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])
    def add(self,state, action, reward, next_state, done):
        # Function to add the experience to the buffer
        state = state[0] if isinstance(state, tuple) else state
        next_state = next_state[0] if isinstance(next_state, tuple) else next_state
        e=self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)
    def sample(self):
        # Function to sample the experience from the buffer
        return random.sample(self.buffer,BATCH_SIZE)
    def __len__(self):
        # Function to get the length of the buffer
        return len(self.buffer)


