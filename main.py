# Import the game
import gym_super_mario_bros

# Import the Joypad wrapper
from nes_py.wrappers import JoypadSpace

# Import the SIMPLIFIED controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Pre-process the environment
# Import GrayScaling wrapper
from gym.wrappers import GrayScaleObservation

# Import Vectorization wrappers
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

# Import Matplotlib to show the impact of frame stacking
from matplotlib import pyplot as plt

# 1. Create the base environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
# 2. Simplify the controls
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# 3. Wrap in grayscale
env = GrayScaleObservation(env, keep_dim=True)
# 4. Wrap in Dummy enviorment
env = DummyVecEnv([lambda: env])
# 5. Wrap in FrameStack wrapper
env = VecFrameStack(env, 4, channels_order='last')

# Import os for file management
import os

# Import PPO for algorithm
from stable_baselines3 import PPO

# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback


class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True


# Create a folder to save a model every 100,000 steps
CHECKPOINT_DIR = './train2/'

# Create a log folder where all the log files from tensorboard goes
LOG_DIR = '/logs/'

# Setup model saving callback
callback = TrainAndLoggingCallback(check_freq=100000, save_path=CHECKPOINT_DIR)

# Create the AI model
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.0000001, n_steps=512)

# Training our model
model.learn(total_timesteps=1000000, callback=callback)

# Running the game with the AI model
state = env.reset()

while True:
    action, _state = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()

