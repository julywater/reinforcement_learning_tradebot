import pandas as pd
import numpy as np
import gym
import math
class TradebotEnvironment(gym.Env):  
    metadata = {'render.modes': ['human']}   
    def __init__(self):
        self.actions = [-1,0,1] # sell, sit, buy
        self.data_features = None # holds feature names
        self.state_size = None #input state size
        self.action_size = len(self.actions) #action size
        self.df = None #data frame which holds data features and close price
        self.initial_account = 100 # dollars
        self.offset = 20 #any episode does not start before the offset
        self.rand_start_index = 0 #random starting index of the episode
        self.pos = np.array([0]) # position: 1 - traded, 0 - no trade
        self.account = self.initial_account # running account
        self.total = 0 # total number of coins
        self.done = 0 # flag indicating the end of the simulation
        self.sim_len = 0 #simulation length
        self.t = 0 #time index
        self.gain = 0 # running profit
        self.episode = 0 #episode number
        self.episode_old = 0 #previous episode number
        self.trades = 0 # number of trades
        self.observation_space = None #observation space
        self.input_shape = None #input shape for the deep network
        self.trade_cost = 0 #trading cost
        self.current_index = 0 #current time index
        self.is_random = True #random start


    def init_file(self, file_name, data_features, trade_cost = 0.5, is_random = True):
        self.is_random = is_random
        self.data_features = data_features
        self.state_size = len(self.data_features) + 1
        self.df = pd.read_csv(file_name)
        self.data_len = self.df.shape[0]
        self.sim_len = self.data_len 
        self.input_shape = self.__get_state().shape
        self.observation_space = self.__get_state()
        self.trade_cost = trade_cost
        return
        
    def step(self, action_id):

        #if an illegal action is taken, the correct it
        action_id = self.correct_action(action_id)
        #update pos according to action
        self.pos[0] = self.pos[0] + self.actions[action_id] 

        #get current and next indexes
        self.current_index = self.rand_start_index + self.t
        next_index = self.current_index + 1

        #retrieve next state line from the data frame
        next_state_line = self.df[self.data_features].iloc[next_index]

        #get current and next prices
        price = self.df.iloc[self.current_index]['close']
        price_next = self.df.iloc[next_index]['close']

        #update account and total number of coins according to action
        if (self.actions[action_id] == 1): # if action is buy
            self.total = self.total + self.account / price
            self.account = 0
        elif (self.actions[action_id] == -1): #if action is sell
            self.account = self.account + price * self.total
            self.total = 0

        # this variable keeps whether any buy or sell action is taken
        action_taken = np.abs(self.actions[action_id])
        
        #update action history
        self.df.at[self.current_index, 'action']= self.actions[action_id]
        
        #calculate current asset
        asset = self.__calculate_asset(price)

        #gain is the profit from the beginning
        self.gain = (asset - self.initial_account)/self.initial_account * 100 
        
        #update profit history
        self.df.at[self.current_index, 'profit']= self.gain
        
        #----- immediate reward calculation algorithm ---------
        reward = 0
        if (self.actions[action_id] == 1):
            reward = math.log(price_next/price)
        elif (self.actions[action_id] == -1):
#            reward = -1.0 * (price_next - price)
             reward=0
        else:
            if self.pos[0] == 0:
    #            reward = -1.0 * (price_next - price)
                 reward=0
            else:
                reward = math.log(price_next/price)
#        reward = reward / price

#        reward_coeff = 100 #percent

#        reward = reward_coeff*reward
#        reward=math.log(reward)
        #---------------------------------------------------

        #increment number of trades if any buy or sell action is taken
        self.trades = self.trades + action_taken

        #discount reward by trade cost id any buy or sell action is taken
        reward = reward - action_taken * self.trade_cost

        #increment time 
        self.t = self.t + 1
        done = False
        if self.current_index == self.sim_len - 2:
            done = True
        
        #construct next state (we also add our postion (pos: 0 or 1) to the state)
        next_state = np.append(next_state_line.to_numpy(),self.pos,axis=None).reshape(1,self.state_size)

        return next_state, reward, done, {}
 
    def reset(self):
        if self.is_random:
            #episode starts between offset and 100 (upper limit)
#            self.rand_start_index = np.random.randint(self.offset,1200)
            self.rand_start_index=(self.episode*262)%(self.sim_len-100)+np.random.randint(self.offset,100)
            print('start index: {}'.format(self.rand_start_index))
        else:
            self.rand_start_index = 0
        self.pos = np.array([0])
        self.account = self.initial_account
        self.total = 0
        self.done = 0
        self.sim_len = self.data_len 
        self.t = 0
        self.gain = 0
        self.trades = 0
        self.episode = self.episode + 1 #increment episode number
        self.current_index = 0
        self.df['action'] = 0 #reset action history
        self.df['profit'] = 0 #reset profit history
        return self.__get_state()
 
    def render(self, mode='human', close=False):
        if self.current_index % 100 == 0:
            if self.episode_old != self.episode:
                self.episode_old = self.episode 
                print('--------------v0----------------')
            print("episode: {}, time: {}, gain: {:.2f}, trades: {}"
                        .format(self.episode, self.current_index, self.gain, self.trades))

    def __calculate_asset(self, price):
        if self.pos[0] == 0:
            return self.account
        else:
            return self.account + self.total * price

    def __get_state(self):
        state_line = self.df[self.data_features].iloc[self.rand_start_index + self.t]

        state = np.append(state_line.to_numpy(),self.pos,axis=None).reshape(1,self.state_size)

        return state

    def correct_action(self,action_id):
        #correction
        if self.pos[0] == 0 and (self.actions[action_id] == -1):
            action_id = 1
        elif self.pos[0] == 1 and (self.actions[action_id] == 1):
            action_id = 1
        return action_id