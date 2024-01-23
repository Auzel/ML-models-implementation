import random
import tqdm
import numpy as np
import matplotlib.pyplot as plt



class Agent:
    def __init__(self, state_space, action_space):
        """
        Initialize table of Q-values to all zeros.
        """
        self.state_space = state_space
        self.action_space = action_space
        self.num_states = len(self.state_space)
        self.num_actions = len(self.action_space)
        self.q_table = np.zeros([self.num_states, self.num_actions])

    def act(self, state, epsilon=0, train=False):
        if train:
            return self._act_train(state, epsilon)
        else:
            return self._act_eval(state)

    def _act_eval(self, state):
        """
        Part (a) Your Code Here. Complete this function.
        Implement action selection in evaluation.
        :param state: the index of a state in the q_table
        :return: the index of an action in the q_table
        """
        return np.argmax(self.q_table[state])
        



    def _act_train(self, state, epsilon):
        """
        Part (b) Your Code Here. Complete this function.
        Please implement epsilon-greedy strategy for action selection in training.
        :param state: the index of a state in the q_table
        :param epsilon: a float number
        :return: the index of an action in the q_table
        """
      
        state_vector_actions=self.q_table[state]
        a_with_max_Q = np.argmax(state_vector_actions)
        a_from_uniform_distrib = np.random.randint(0,np.size(state_vector_actions))
        return np.random.choice([a_with_max_Q,a_from_uniform_distrib],p=[1-epsilon,epsilon])



    def update(self, state, action, reward, next_state, alpha, gamma):
        """
        Part (c) Your Code Here. Complete this function.
        Implement Q-value update using the Bellman update to update self.q_table[state, action].
        :param state: the index of a state in the q_table
        :param action: the index of an action in the q_table
        :param reward: a float number
        :param next_state: the index of a state in the q_table
        :param alpha: a float number
        :param gamma: a float number
        """
        self.q_table[state,action]=(1-alpha)*self.q_table[state,action] + alpha*(reward+gamma*np.max(self.q_table[next_state]))


def train_agent(env, agents, random_agent, epochs=1000, alpha=0.1, gamma=0.9, epsilon=0.1, plot_interval=25):
    agent_1_vs_2_win = []
    agent_1_vs_random_win = []
    agent_2_vs_random_win = []
    for epoch in tqdm.tqdm(range(epochs)):
        env.reset_state()
        done = False
        state_list = []
        action_list = []
        reward_list = []
        initial_player = np.random.randint(2)  # random first player
        player = initial_player

        # evaluate agents
        if epoch % plot_interval == 0:
            game_result = eval_agent(env, agents[0], agents[1], num_trials=100)
            agent_1_vs_2_win.append(game_result)
            game_result = eval_agent(env, agents[0], random_agent, num_trials=100)
            agent_1_vs_random_win.append(game_result)
            game_result = eval_agent(env, agents[1], random_agent, num_trials=100)
            agent_2_vs_random_win.append(game_result)

        # collect the episode
        while not done:
            agent = agents[player]
            state_list.append(env.state)
            action = agent.act(env.state, epsilon, train=True)
            reward, done = env.step(action)
            action_list.append(action)
            reward_list.append(reward)
            player = (player + 1) % 2  # alternate player
        state_list.append(env.state)

        """
        Part (d) Your Code Here. Complete the RL training.
        Use the state_list, action_list and reward_list and the update() function of the agents. 
        """

        pass

        """
        No more your code below this line.
        """

    print("Training finished.\n")
    return agent_1_vs_2_win, agent_1_vs_random_win, agent_2_vs_random_win


class GameEnv:
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.num_states = len(self.state_space)
        self.num_actions = len(self.action_space)
        self.state = 0

    def reset_state(self):
        self.state = 0

    def step(self, action):
        state_step_size = self.action_space[action]
        if self.state + state_step_size >= self.num_states:
            # losing step
            reward = - 1
            next_state = self.num_states - 1
            done = True
        elif self.state + state_step_size == self.num_states - 1:
            # winning step
            reward = 1
            next_state = self.num_states - 1
            done = True
        else:
            # normal step
            reward = 0
            next_state = self.state + state_step_size
            done = False
        self.state = next_state
        return reward, done


class RandomAgent:
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.num_states = len(self.state_space)
        self.num_actions = len(self.action_space)

    def act(self, state, epsilon=0, train=False):
        for i in range(self.num_actions):
            if state + self.action_space[i] == self.num_states - 1:
                return i
        return np.random.randint(self.num_actions)

    def update(self, state, action, reward, next_state, alpha, gamma):
        pass


def eval_agent(env, agent, baseline_agent, num_trials=100):
    # evaluate the winning rate against the RandomAgent
    agents = [agent, baseline_agent]
    agent_win_counter = 0
    for trial in range(num_trials):
        done = False
        env.reset_state()
        player = np.random.randint(2)  # random first player
        while not done:
            agent = agents[player]
            action = agent.act(env.state)
            reward, done = env.step(action)
            if done:
                if (reward == 1 and player == 0) or (reward < 0 and player == 1):
                    agent_win_counter += 1
            player = (player + 1) % 2  # alternate player
    env.reset_state()
    return agent_win_counter / num_trials


def play_with_human(env, agent):
    print("Starting from 0, who gets to {} first is the winner!".format(env.num_states - 1))
    print("Please select from {} to increase the value".format(env.action_space))
    env.reset_state()
    done = False
    player = np.random.randint(2)  # random first player
    action_space_np = np.array(env.action_space)
    if player == 0:
        print('You go first')
    else:
        print('Agent goes first')
    while not done:
        print('\nThe current value is {}. Who gets {} first is the winner!'.format(env.state, env.num_states - 1))
        if player == 0:
            # human player
            action_compete = False
            while not action_compete:
                print('please select from {} to increase the value'.format(env.action_space))
                try:
                    user_selection = int(input())
                    action = np.where(action_space_np == user_selection)[0].item()
                    action_compete = True
                    print('you chose {}'.format(user_selection))
                except ValueError:
                    print('Oops! Invalid input. Try again!')
        else:
            # agent player
            action = agent.act(env.state)
            print('The Agent chooses {}'.format(env.action_space[action]))
        reward, done = env.step(action)
        if done:
            if (reward == 1 and player == 0) or (reward < 0 and player == 1):
                print('YOU WIN!!!!!')
            else:
                print('YOU LOSE....')
        player = (player + 1) % 2  # alternate players


def plot(agent_1_vs_2_win, agent_1_vs_random_win, agent_2_vs_random_win, plot_interval=100):
    num_points = len(agent_1_vs_2_win)
    epoch_list = np.array(range(num_points)) * plot_interval
    plt.plot(epoch_list, agent_1_vs_2_win, label="agent_1_vs_2_win_rate")
    plt.plot(epoch_list, agent_1_vs_random_win, label="agent_1_vs_random_win_rate")
    plt.plot(epoch_list, agent_2_vs_random_win, label='agent_2_vs_random_win_rate')
    plt.ylim(-0.05, 1.05)
    plt.yticks(np.arange(-0.05, 1.05, step=0.05))
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel('Win Rate')
    plt.tick_params(axis='y', which='both', labelleft=True, labelright=True)
    plt.savefig("./train_win_rate.png")
    plt.show()


if __name__ == "__main__":
    # you may want to try different settings for fun
    # but please use the original setting exactly for the homework
    target_num = 21
    state_space = list(range(target_num + 1))
    action_space = [1, 3, 4]
    plot_interval = 25
    num_epochs = 1000
    random.seed(2021)
    np.random.seed(2021)

    env = GameEnv(state_space, action_space)
    agent_1 = Agent(state_space, action_space)
    agent_2 = Agent(state_space, action_space)
    random_agent = RandomAgent(state_space, action_space)
    agents = [agent_1, agent_2]
    agent_1_vs_2_win, agent_1_vs_random_win, agent_2_vs_random_win = train_agent(env, agents, random_agent,
                                                                                 epochs=num_epochs, alpha=0.1,
                                                                                 gamma=0.9, epsilon=0.1,
                                                                                 plot_interval=plot_interval)
    win_rate = eval_agent(env, agent_1, agent_2, num_trials=100)
    if win_rate > 0.5:
        agent = agent_1
    else:
        agent = agent_2

    # plot
    plot(agent_1_vs_2_win, agent_1_vs_random_win, agent_2_vs_random_win, plot_interval=plot_interval)

    # try to play with your agents!
    human_play = True
    while human_play:
        play_with_human(env, agent)
        respond = input('Try Again? y/n: ')
        if respond != 'y' and respond != 'Y' and respond != 'yes' and respond != 'Yes':
            human_play = False
