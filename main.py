from features import FeatureExtractor
from keras_model import create_model
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
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
     max_episode_steps=262,
)

env = make(ENV_NAME)

input_file =  'data/MSFT_1d_train.csv'
output_file = 'data/MSFT_1d_train_fe.csv'
w_file_name = 'data/MSFT_1d.h5f'

feature_extractor = FeatureExtractor(input_file, output_file)
feature_extractor.extract()

feature_list = feature_extractor.get_feature_names()

trade_cost = 0.03
env.init_file(output_file, feature_list, trade_cost)

num_training_episodes = 90
model = create_model(env)
memory = SequentialMemory(limit=5000, window_length=1)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=env.action_size, memory=memory,
               nb_steps_warmup=50, target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mse'])
print(num_training_episodes * env.sim_len)
dqn.fit(env, nb_steps=num_training_episodes * env.sim_len, visualize=True, verbose=0)

# Here, we save the final weights.
dqn.save_weights(w_file_name, overwrite=True)