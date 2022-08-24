import gym
import numpy as np
import copy, json, argparse
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from random import uniform

class ReplayBuffer(object):

    def __init__(self, mem_size, input_shape):
        self.memorySize=mem_size
        self.counter=0

        self.state_transition=np.zeros((self.memorySize, *input_shape), dtype=np.float32)
        self.newstate_transition = np.zeros((self.memorySize, *input_shape), dtype=np.float32)
        self.action_transition = np.zeros((self.memorySize, *input_shape), dtype=np.int32)
        self.reward_transition=np.zeros(self.memorySize, dtype=np.float32)
        self.terminal_transition=np.zeros(self.memorySize, dtype=np.uint8)

    def store_transition(self,state, new_statte, action, reward, done):
        index=self.counter%self.memorySize

        self.state_transition[index]=state
        self.newstate_transition[index] = new_statte
        self.action_transition[index] = action
        self.reward_transition[index] = reward
        self.terminal_transition[index] = done

        self.counter+=1

    def sample_buffer(self, batch_size):
        max_mem=min(self.counter, self.memorySize)
        batch=np.random.choice(max_mem, batch_size, replace=False)

        state=self.state_transition[batch]
        new_state = self.newstate_transition[batch]
        action = self.action_transition[batch]
        reward = self.reward_transition[batch]
        done=self.terminal_transition[batch]

        return state, new_state, action, reward, done

class DuelingDeepQNetwork(nn.Module):

    def __init__(self, lr,   n_action, input_shape,name,  check_dir):
        super(DuelingDeepQNetwork, self).__init__()
        self.check_dir=check_dir
        self.checkpt_file=os.path.join(self.check_dir, name)

        self.L1=nn.Linear(*input_shape, 512)
        self.V=nn.Linear(512,1)
        self.A=nn.Linear(512, n_action)

        self.optimizer=optim.Adam(self.parameters(), lr=lr)
        self.loss=nn.MSELoss()

        self.device=T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        flat1=F.relu(self.L1(state))
        V=self.V(flat1)
        A=self.A(flat1)

        return V, A

    def save_checkpoint(self):
        print('... saving checkpoints ...')
        T.save(self.state_dict(), self.checkpt_file)

    def load_checkpoint(self):
        print('... loading checkpoints ...')
        self.load_state_dict(T.load(self.checkpt_file))

class Agent(object):

    def __init__(self, nUsers, nChannel, gamma, epsilon,lr, n_action, input_dims, mem_size, batch_size, eps_min=0.01, eps_decay=5e-7, replace=1000,  check_dir='tmp\D3QN'):
        self.nUsers=nUsers
        self.nChannel=nChannel
        self.gamma=gamma
        self.epsilon=epsilon
        self.lr=lr
        self.n_action=n_action
        self.input_dims=input_dims
        self.mem_size=mem_size
        self.batch_size=batch_size
        self.eps_min=eps_min
        self.eps_decay=eps_decay
        self.replace_target_cnt=replace
        self.check_dir=check_dir
        self.learn_step_counter=0
        self.action_space=[i for i in range(self.n_action)]

        self.memery = ReplayBuffer(self.mem_size, self.input_dims)
        self.q_eval = DuelingDeepQNetwork(self.lr, self.n_action, input_shape=self.input_dims, name='D3QN q_eval',
                           check_dir=self.check_dir)
        self.q_next = DuelingDeepQNetwork(self.lr, self.n_action, input_shape=self.input_dims, name='D3QN q_next',
                           check_dir=self.check_dir)

    def choose_action(self,observation):
        actions=[]
        for i in range(self.nUsers):
            if np.random.random()> self.epsilon:
                state=T.tensor([observation], dtype=T.float).to(self.q_eval.device)
                _,advantage=self.q_eval.forward(state)
                action=T.argmax(advantage).item()
                actions.append(action)
            else:
                action=np.random.choice(self.action_space)
                actions.append(action)

        return actions

    def store_transition(self, state, new_state, action, reward, done):
        self.memery.store_transition(state, new_state, action, reward, done)

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt==0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        if self.epsilon>self.eps_min:
            self.epsilon=self.epsilon-self.eps_decay
        else:
            self.epsilon=self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memery.counter < self.batch_size:
            return
        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()

        state, new_state, action, reward, done = self.memery.sample_buffer(self.batch_size)
        states = T.tensor(state).to(self.q_eval.device)
        new_states = T.tensor(new_state).to(self.q_eval.device)
        actions = T.tensor(action,dtype=T.int32).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)

        indices = np.arange(self.batch_size)  ##### Critical and necessary

        Vs, As = self.q_eval.forward(states)
        Vs_new, As_new = self.q_next.forward(new_states)
        Vs_eval, As_eval = self.q_eval.forward(new_states)

        q_pred = T.add(Vs, (As - As.mean(dim=1, keepdim=True)))[indices, actions]  # for sampled actions
        q_next = T.add(Vs_new, (As_new - As_new.mean(dim=1, keepdim=True)))  # for all actions
        q_eval = T.add(Vs_eval, (As_eval - As_eval.mean(dim=1, keepdim=True)))

        max_actions = T.argmax(q_eval, dim=1)
        q_next[dones]=0.0
        q_target=rewards+self.gamma*q_next[indices,max_actions]

        loss=self.q_eval.loss(q_target,q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter+=1

        self.decrement_epsilon()

class Environ:

    def __init__(self, nUsers,nChannel, Noise, Pmax, Rmin, negative_cost, circuitPower, BW):
        self.nUsers = nUsers
        self.nChannel=nChannel
        self.Noise=Noise
        self.Pmax=Pmax
        self.Rmin=Rmin
        self.negative_cost=negative_cost
        self.circuitPower=circuitPower
        self.BW=BW

        self.state_dim= nUsers
        self.action_dim=nChannel
        self.bs = complex((500 / 2), (500 / 2))
        self.S=(np.zeros(self.nUsers)).reshape(-1)

    def Location(self):
        rx = uniform(0, 500)
        ry = uniform(0, 500)
        Loc = complex(rx, ry)
        return Loc

    def PathGain(self,Loc):
        d = abs(Loc - self.bs)
        d = d ** (-3)
        u = np.random.rand(1, 1)
        sigma = 1
        x = sigma * np.sqrt(-2 * np.log(u))
        h = d * x
        return h

    def reset(self):  # Reset the states
        s=np.zeros(self.nUsers)
        return s.reshape(-1)

    def RecievePower(self,UsersLoc):
        H=self.PathGain(UsersLoc)
        UsersRecievePower = H*self.Pmax
        return UsersRecievePower

    def TotalRate(self, actionRB):
        interference = np.zeros(self.nUsers, dtype=float)+self.Noise
        RecievePower = np.zeros(self.nUsers, dtype=float)
        SINR = np.zeros(self.nUsers, dtype=float)
        Rate = np.zeros(self.nUsers, dtype=float)
        for i in range(self.nUsers):
            Loc_i = self.Location()
            for j in range(self.nUsers):
                if j!=i and actionRB[i] ==actionRB[j] :
                    Loc_j = self.Location()
                    RecievePower[j] = self.RecievePower(Loc_j)
                    interference [i]= interference [i]+ RecievePower[j]
                else:
                    interference[i]= interference[i]
            RecievePower[i] = self.RecievePower(Loc_i)
            SINR[i] = RecievePower[i] / interference[i]
            Rate[i] =self.BW*( np.log2( SINR[i]))
        return Rate

    def computeQoS(self,actionRB):
        TotalRate=self.TotalRate(actionRB)
        QoS= []
        for i in range(self.nUsers):
            if TotalRate[i] >=self.Rmin:
                QoS.append(1.0)
            else:
                QoS.append(0.0)
        return QoS

    def ComputeState(self,actionRB):
        QoS=self.computeQoS(actionRB)
        S = np.zeros( self.nUsers)
        for i in range(self.nUsers):
            S[i]=QoS[i]
        self.S=S
        return self.S.reshape(-1)

    def Reward(self,actionRB):
        Rate = self.TotalRate( actionRB)
        QoS=self.computeQoS(actionRB)
        Satisfied_Users = sum(QoS)
        TotalRate = 0.0
        TotalPower = self.circuitPower
        for i in range(self.nUsers):
            TotalRate = TotalRate + Rate[i]
            TotalPower = self.Pmax
        if Satisfied_Users == self.nUsers:
            reward = TotalRate / TotalPower
            done=True
        else:
            reward = self.negative_cost
            done=False
        # print('Satisfied_Users= ', Satisfied_Users)
        return reward, done

    def step(self,actionRB):
        next_s = self.ComputeState(actionRB)
        r, d = self.Reward(actionRB)
        done = False
        info = None
        if d==True:
            done=True
        return next_s, r, done, info

def plot_learning_curve(x, scores,  filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    # ax2=fig.add_subplot(111, label="2")



    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax.plot(x,running_avg , color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Score", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)

nUsers=10
nChannel=30
Noise=1e-14
Pmax=0.01
Rmin=1e+6
negative_cost=-1
circuitPower=50*10**3
BW=180*10**3
max_nsteps=100

if __name__=='__main__':

    env = Environ(nUsers,nChannel, Noise, Pmax, Rmin, negative_cost, circuitPower, BW)

    episodes = 1
    load_checkpoint = False

    agent = Agent(nUsers, nChannel, gamma=0.99, epsilon=1.0, lr=5e-4,
                      n_action=nChannel, input_dims=[nUsers], mem_size=1000000, batch_size=64,
                      eps_min=0.01, eps_decay=1e-3, replace=100)

    if load_checkpoint:
        agent.load_models()

    filename = 'Resources.jpg'
    scores, eps_history = [], []

    for i in range(episodes):
        done = False
        observation = env.reset()
        score = 0
        nstep=0
        while  nstep<=max_nsteps:
            nstep+=1
            action = agent.choose_action(observation)
            new_observation, reward, done, infor = env.step(action)
            score += reward
            agent.store_transition(observation, new_observation, action, reward, int(done))
            agent.learn()
            observation = new_observation


        scores.append(score)
        avg_score = np.mean(scores[-100:])

        print('episode: ', i, 'score%.1f' % score, 'average score%.1f' % avg_score, 'epsilon %.2f' % agent.epsilon)

        if i>10 and i%10==0:
            agent.save_models()

        eps_history.append(agent.epsilon)

    x=[i+1 for i in range(episodes)]
    plot_learning_curve(x, scores, filename)

