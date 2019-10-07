import random
import numpy as np
from tqdm import tqdm
import json
from gym_tictactoe.env import TicTacToeEnv, agent_by_mark, next_mark,\
     after_action_state, check_game_status, O_REWARD, X_REWARD

#from human_agent import HumanAgent
from base_agent import BaseAgent
from td_agent import TDAgent

MAX_EPISODE = 20000
EPSILON = 0.5
MC_MODEL_FILE = 'best_mconp_agent.dat'

Q_Table = {}

class OnPolicyMCAgent(object):
    def __init__(self, mark, epsilon, test=0):
        self.mark = mark
        self.epsilon = epsilon
        self.trans_list = []
        self.test = test
        self.state_count = {}
        self.epolicy = {}
        self.total_actions = 9
        self.orig_actions = []

    def act(self,state,ava_actions):
        #state = state[0]
        prob_list = []
        prob_sum = 0
        if state not in Q_Table:
            #print('Bye')
            return np.random.choice(ava_actions)

        #returning actions while testing
        if self.test == 1:
            #print('Hi')
            if self.mark == 'O':
                return np.argmax(Q_Table[state])
            else:
                return np.argmin(Q_Table[state])

        for action in ava_actions:
            prob_list.append(self.epolicy[state][action])
            prob_sum+=self.epolicy[state][action]

        for action in range(len(prob_list)):
            prob_list[action] += (1-prob_sum)/len(prob_list)

        action = action = np.random.choice(ava_actions,p=prob_list)

        return action

    def update_policy(self,state):
        if self.mark == 'O':
            a_star = np.argmax(Q_Table[state])
        else:
            a_star = np.argmin(Q_Table[state])
        for action in range(self.total_actions):
            self.epolicy[state] = [self.epsilon/self.total_actions]*\
                                                        self.total_actions
        self.epolicy[state][a_star] += 1-self.epsilon

    def update(self):
        trans_size = len(self.trans_list)-1
        return_val = 0
        for transition in range(trans_size,-1,-1):
            state,action,reward = self.trans_list[transition]
            #state = state[0]
            if state not in Q_Table:
                Q_Table[state] = [0]*self.total_actions
            if (state,action) not in self.state_count:
                self.state_count[(state,action)] = 1
            else:
                self.state_count[(state,action)] += 1
            return_val= return_val + reward

            diff = return_val - Q_Table[state][action]
            Q_Table[state][action] +=  diff/self.state_count[(state,action)]
            self.update_policy(state)


def save_model(save_file):
    with open(save_file, 'wt') as f:
        # write model info
        info = dict(type="MC")
        # write state values
        f.write('{}\n'.format(json.dumps(info)))
        for state, value in Q_Table.items():
            #print(state,value)
            f.write('{}\t{}\n'.format(state, str(value)))


def load_model(filename):
    with open(filename, 'rb') as f:
        # read model info
        info = json.loads(f.readline().decode('ascii'))
        for line in f:
            elms = line.decode('ascii').split('\t')
            state = eval(elms[0])
            val = eval(elms[1])
            Q_Table[state] = val
    return info

def learn(env):
    max_episode = MAX_EPISODE
    epsilon = EPSILON
    agents = [OnPolicyMCAgent('X', epsilon),
              OnPolicyMCAgent('O', epsilon)]
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
            #agent.update(state, new_state)
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
              OnPolicyMCAgent('X', 0, 1)]

    start_mark = 'X'
    test_cases = 10
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
                env.show_result(True, mark, reward)
                break
            else:
                _, mark = state

        # rotation start
        start_mark = next_mark(start_mark)
        test_cases-=1


if __name__ == '__main__':
    env = TicTacToeEnv()
    learn(env)
    play_base(env)
