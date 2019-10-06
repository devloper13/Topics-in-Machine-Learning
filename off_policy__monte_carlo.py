import random
import numpy as np
from tqdm import tqdm
import json
from gym_tictactoe.env import TicTacToeEnv, agent_by_mark, next_mark,\
     after_action_state, check_game_status, O_REWARD, X_REWARD

from base_agent import BaseAgent
MAX_EPISODE = 600000
EPSILON = 0.5
MC_MODEL_FILE = 'best_mcoffp_agent.dat'

Q_Table = {}
ppolicy = {}

class OffPolicyMCAgent(object):
    def __init__(self, mark, epsilon, test=0):
        self.mark = mark
        self.epsilon = epsilon
        self.trans_list = []
        self.test = test
        self.C = {}
        self.bpolicy = {}
        self.total_actions = 9
        self.orig_actions = []
        self.gamma = 1

    def act(self,state,ava_actions):
        prob_list = []
        prob_sum = 0
        if state not in Q_Table:
            return np.random.choice(ava_actions)

        if self.test == 1:
            #print('Hi')
            return ppolicy[state]

        for action in ava_actions:
            prob_list.append(self.bpolicy[state][action])
            prob_sum+=self.bpolicy[state][action]

        for action in range(len(prob_list)):
            prob_list[action] += (1-prob_sum)/len(prob_list)

        action = np.random.choice(ava_actions,p=prob_list)

        return action

    def update_policy(self,state):
        if self.mark == 'O':
            a_star = np.argmax(Q_Table[state])
        else:
            a_star = np.argmin(Q_Table[state])

        for action in range(self.total_actions):
            self.bpolicy[state] = [self.epsilon/self.total_actions]*\
                                                        self.total_actions
        self.bpolicy[state][a_star] += 1-self.epsilon

    def update(self):
        trans_size = len(self.trans_list)-1
        return_val = 0
        W = 1
        for transition in range(trans_size,-1,-1):
            state,action,reward = self.trans_list[transition]
            if state not in Q_Table:
                Q_Table[state] = [0]*self.total_actions
                self.bpolicy[state] = [1/self.total_actions]*self.total_actions
                #self.update_policy(state)

            if (state,action) not in self.C:
                self.C[(state,action)] = 0
            self.C[(state,action)] += W

            return_val= self.gamma * return_val + reward

            diff = return_val - Q_Table[state][action]
            Q_Table[state][action] +=  diff * (W/self.C[(state,action)])

            if self.mark == 'O':
                ppolicy[state] = np.argmax(Q_Table[state])
            else:
                ppolicy[state] = np.argmin(Q_Table[state])

            if action != ppolicy[state]:
                break

            W = W*(1/self.bpolicy[state][action])


def save_model(save_file):
    with open(save_file, 'wt') as f:
        # write model info
        info = dict(type="MC")
        # write state values
        f.write('{}\n'.format(json.dumps(info)))
        for state, value in ppolicy.items():
            #print(state,value)
            f.write('{}\t{}\n'.format(state, value))


def load_model(filename):
    with open(filename, 'rb') as f:
        # read model info
        info = json.loads(f.readline().decode('ascii'))
        for line in f:
            elms = line.decode('ascii').split('\t')
            state = eval(elms[0])
            val = eval(elms[1])
            ppolicy[state] = val
    return info

def learn(env):
    max_episode = MAX_EPISODE
    epsilon = EPSILON
    agents = [OffPolicyMCAgent('O', epsilon),
              OffPolicyMCAgent('X', epsilon)]
    agents[0].orig_actions = env.available_actions()
    agents[1].orig_actions = env.available_actions()

    start_mark = 'O'

    #iterating through episodes
    for episode in tqdm(range(max_episode)):

        #env.show_episode(False,episode+1)
        agents[0].trans_list = []
        agents[1].trans_list = []
        env.set_start_mark(start_mark)
        state = env.reset()
        _,mark = state
        done = False
        while not done:
            agent = agent_by_mark(agents, mark)
            ava_actions = env.available_actions()
            env.show_turn(False,mark)
            action = agent.act(state,ava_actions)
            next_state, reward, done, _ = env.step(action)
            agent.trans_list.append((state,action,reward))

            state = next_state
            _, mark = state


        agents[0].update();
        agents[1].update();

        if done:
            env.show_result(False,mark,reward)

        start_mark = next_mark(start_mark)
    save_model(MC_MODEL_FILE)

def play_base(env):
    load_model(MC_MODEL_FILE)
    agents = [BaseAgent('O'),
              OffPolicyMCAgent('X', 0, 1)]

    start_mark = 'O'
    test_cases = 1000
    win1, win2 = 0,0
    while test_cases:
        env.set_start_mark(start_mark)
        state = env.reset()
        _,mark = state
        done = False
        while not done:
            agent = agent_by_mark(agents, mark)
            ava_actions = env.available_actions()
            env.show_turn(False,mark)
            action = agent.act(state,ava_actions)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            if done:
                #env.show_result(True, mark, reward)
                if reward != 0 and mark == agents[0].mark:
                    win1 += 1
                elif reward != 0 and mark == agents[1].mark:
                    win2 += 1
                break
            else:
                _, mark = state

        # rotation s tart
        #start_mark = next_mark(start_mark)
        test_cases-=1
    print(agents[0].mark, win1, agents[1].mark, win2)

if __name__ == '__main__':
    env = TicTacToeEnv()
    learn(env)
    play_base(env)
