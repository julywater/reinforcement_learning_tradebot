from features import FeatureExtractor
from keras_model import create_model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy
import gym
from gym.envs.registration import register, make
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
ENV_NAME = 'TradebotEnvironment-v0'

if ENV_NAME in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs[ENV_NAME]
    
register(
     id=ENV_NAME,
     entry_point='trade_env:TradebotEnvironment',
     max_episode_steps=10000,
)

env = make(ENV_NAME)

input_file = 'data/MSFT_1d_test.csv'
output_file = 'data/MSFT_1d_test_fe.csv'
w_file_name = 'data/MSFT_1d.h5f'

feature_extractor = FeatureExtractor(input_file, output_file)
feature_extractor.extract()

feature_list = feature_extractor.get_feature_names()

trade_cost = 0.03
env.init_file(output_file, feature_list, trade_cost, False)


model = create_model(env)
memory = SequentialMemory(limit=5000, window_length=1)
policy = GreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=env.action_size, memory=memory, 
               nb_steps_warmup=50, target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mse'])

dqn.load_weights(w_file_name)

dqn.test(env, nb_episodes=1, action_repetition=1, callbacks=None,
     visualize=True, nb_max_episode_steps=None,
     nb_max_start_steps=0, start_step_policy=None, verbose=1)

fig = plt.figure()
gs = gridspec.GridSpec(2, 1, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(env.df['close'],'-b',linewidth = 0.5)
ax1.plot(env.df['close'][env.df['action']>0],'og', label='Buy',markersize = 10)
ax1.plot(env.df['close'][env.df['action']<0],'or', label ='Sell',markersize = 10)
ax1.set_xlabel('Time')
ax1.set_ylabel('Price')
ax1.legend()


start_price=env.df['close'].iloc[0]

ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(env.df['profit'][:-1],'-r',linewidth = 1,label='trade_bot')
ax2.plot(100*(env.df['close']/start_price-1),'-b',linewidth = 1,label='buy&hold')
ax2.legend()
ax2.set_xlabel('Time')
ax2.set_ylabel('Profit (%)')

plt.tight_layout()
plt.show()
print(env.df['profit'].iloc[-2])
print("buy&Hold Profit:",100*(env.df['close'].iloc[-1]/env.df['close'].iloc[0]-1))