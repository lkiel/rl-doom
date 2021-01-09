import stable_baselines3
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy

from models.cnn import CNNFeatureExtractor

MODELS = {'PPO': PPO, 'DQN': DQN}

POLICIES = {'PPO': ActorCriticCnnPolicy, 'DQN': stable_baselines3.dqn.CnnPolicy}

NETS = {'CNNFeatureExtractor': CNNFeatureExtractor}
