import random
import numpy as np
from tqdm import tqdm
from gym_tictactoe.env import TicTacToeEnv, agent_by_mark, next_mark,\
     after_action_state, check_game_status, O_REWARD, X_REWARD

MAX_EPISODE = 17000
EPSILON = 0.08
MODEL_FILE = 'best_mconp_agent.dat'

class OnPolicyMCAgent(object):
    def __init__(self, mark, epsilon):
        self.ava_actions = []
        self.mark = mark
        self.epsilon = epsilon
        self.trans_list = []
        self.Q_Table = {}
        self.state_count = {}
        self.epolicy = {}
        self.total_actions = 9
        self.orig_actions = []

    def act(self,state):
        state = state[0]
        prob_list = []
        prob_sum = 0
        if state not in self.epolicy:
            self.Q_Table[state] = [0]*self.total_actions
            self.update_policy(state)

        for action in self.ava_actions:
            prob_list.append(self.epolicy[state][action])
            prob_sum+=self.epolicy[state][action]

        for action in range(len(prob_list)):
            prob_list[action] += (1-prob_sum)/len(prob_list)

        action = action = np.random.choice(self.ava_actions,p=prob_list)
        #action = np.random.choice(self.orig_actions,p=self.epolicy[state])

        #while action not in self.ava_actions:
        #    action = np.random.choice(self.orig_actions,p=self.epolicy[state])
        #if action not in self.ava_actions:
        #    action = np.random.choice(self.ava_actions)
        return action

    def update_policy(self,state):
        a_star = np.argmax(self.Q_Table[state])
        for action in range(self.total_actions):
            self.epolicy[state] = [self.epsilon/self.total_actions]*\
                                                        self.total_actions
        self.epolicy[state][a_star] += 1-self.epsilon

    def update(self):
        trans_size = len(self.trans_list)-1
        return_val = 0
        for transition in range(trans_size,0,-1):
            state,action,reward = self.trans_list[transition]
            if (state,action) not in self.state_count:
                self.state_count[(state,action)] = 1
                self.Q_Table[state] = [0]*self.total_actions
            else:
                self.state_count[(state,action)] += 1
            return_val= return_val + reward

            diff = return_val - self.Q_Table[state][action]
            self.Q_Table[state][action] +=  diff/self.state_count[(state,action)]
            self.update_policy(state)


def learn():
    max_episode = MAX_EPISODE
    epsilon = EPSILON
    env = TicTacToeEnv()
    agents = [OnPolicyMCAgent('O', epsilon),
              OnPolicyMCAgent('X', epsilon)]
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
            agent.ava_actions = env.available_actions()
            env.show_turn(False,mark)
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            #agent.update(state, new_state)
            agent.trans_list.append((state,action,reward))

            state = next_state
            _, mark = state


        agents[0].update();
        agents[1].update();

        if done:
            env.show_result(False,mark,reward)
            #set_state_value(state,reward)  # TODO: implement function

        start_mark = next_mark(start_mark)


if __name__ == '__main__':
    learn()
