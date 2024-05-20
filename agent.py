import numpy as np
import matplotlib.pyplot as plt


def egreedy(v,e=0.95):
    NA = len(v)
    b = np.isclose(v,np.max(v))
    no = np.sum(b)
    if no<NA:
        p = b*e/no+(1-b)*(1-e)/(NA-no)
    else:
        p = b/no

    return int(np.random.choice(np.arange(NA),p=p))

class Agent:
    def __init__(self, id, grid_size,n_apples,n_agents, NA = 6, alpha = 0.1, gamma = 0.9, agentType = 'independent'):
        self.alpha = alpha
        self.gamma = gamma
        self.id = id
        self.agentType = agentType
        self.NL = grid_size * grid_size
        self.NA = NA
        self.grid_size = grid_size
        self.n_apples = n_apples
        self.n_agents = n_agents
        
        if self.agentType == 'independent':
            self.Q = np.ones(((grid_size ** 2) ** (n_agents + n_apples),NA))*2
        elif self.agentType == 'observ':
            self.Q = np.ones((self.NL*self.NL,NA))*2
        elif self.agentType == 'central':
            self.Q = np.ones((self.NL*self.NL,NA*NA))*2
        elif self.agentType == 'JALAM':
            self.Q = np.ones((self.NL*self.NL,NA,NA))*2
            self.C = np.zeros((self.NL*self.NL,NA,NA))+0.001

    def __str__(self):

        return f"#{self.id} Qshape{self.Q.shape}"

    # this function takes the whole environment state and projects it in the state that each type of agent has access
    def observ2state(self, x):
        states = []
        x = list(map(int, x))
        for i in range(0,len(x),3):
            if x[i] == -1:
                self.n_apples -= 1
            else:
                state  = np.ravel_multi_index(x[i:i+2], (self.grid_size, self.grid_size))
                states.append(state)

        if self.agentType == 'independent':
            shape = tuple([self.NL, self.NL, self.NL, self.NL])
            print(states)
            return  np.ravel_multi_index(states,shape)
        elif self.agentType == 'observ' or self.agentType == 'central' or self.agentType == 'JALAM':
            shape = tuple(self.NL - i for i in range(self.n_apples+self.n_agents))
            return np.ravel_multi_index(states,shape)
        
    # this function is the learning update after any iteration with the environment, it gets
    # x current state
    # nx next state
    # a selected actions
    # r rewards obtained
    def update(self,x,nx,a,r):
        xi = self.observ2state(x)
        nxi = self.observ2state(nx)

        x = list(map(int, x))
        if self.agentType == 'central':
            ai = np.ravel_multi_index(a,[self.NA,self.NA])
            self.Q[xi,ai] += self.alpha * (np.sum(r) + self.gamma * np.max(self.Q[nxi,:]) - self.Q[xi,ai])
        elif self.agentType == 'JALAM':
            self.C[xi,a[self.id],a[1-self.id]] += 1
            self.Q[xi,a[self.id],a[1-self.id]] += self.alpha * (r[self.id] + self.gamma * np.max(self.Q[nxi,:]) - self.Q[xi,a[self.id],a[1-self.id]])
        else:
            self.Q[xi,a[self.id]] += self.alpha * (r[self.id] + self.gamma * np.max(self.Q[nxi,:]) - self.Q[xi,a[self.id]])
        return self.Q[x,:]

    # choosing the action to make in a given state x
    def chooseAction(self,x,e):
        xi = self.observ2state(x)
        if type(e) is float:
            if self.agentType == 'central':
                A = egreedy(self.Q[xi,:],e=e)
                a = np.unravel_index(A,[self.NA,self.NA])
                return a[self.id]
            elif self.agentType == 'JALAM':
                policyB = np.sum(self.C[xi,:,:],axis = 1)
                policyB = policyB/np.sum(policyB)

                bestreponseA = self.Q[xi,:,:] @ policyB.T

                actA = np.argmax(bestreponseA)
                return actA
            else:
                return egreedy(self.Q[xi,:],e=e)
            




