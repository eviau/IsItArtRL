import gym
import logging.config

from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np

from PIL import Image
import random


class isitartrlEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.__version__ = "1"
        logging.info("isitartrlEnv - Version {}".format(self.__version__))
        self.TOTAL_TIME_STEPS = 1000000
        self.curr_step = -1
        self.action_space = spaces.MultiDiscrete(
            [100, 100, 255, 255, 255])
        low = np.array([0.0])
        high = np.array([self.TOTAL_TIME_STEPS])

        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.curr_episode = -1
        self.action_episode_memory = []

        self.current_painting = np.zeros((100, 100, 3), dtype=np.uint8)
        self.is_picture_art = False
        image = Image.fromarray(self.current_painting)
        image.show()

    def step(self, action):

        if self.is_picture_art:
            raise RuntimeError("It is art ! Marvelous!")
        self.curr_step += 1
        self._take_action(action)
        reward = self._get_reward()
        ob = self._get_state()
        return ob, reward, self.is_picture_art, {}

    def _take_action(self, action):
        self.action_episode_memory[self.curr_episode].append(action)

        x = action[0]
        y = action[1]
        r = action[2] + 255
        b = action[3] + 255
        g = action[4] + 255

        self.current_painting[x, y] = [r, b, g]
        print(action)

        self.is_picture_art = self._eval_is_pict_art()

    def _eval_is_pict_art(self):
        print("is this picture art ?")
        image = Image.fromarray(self.current_painting)
        image.show()

        temp = input('Enter 1 if it art, 0 if not:')
        temp = random.randint(0, 1)
        if (temp == 1):
            self.is_picture_art = True
        elif (temp == 1):
            self.is_picture_art = False

    def _get_reward(self):
        if self.is_picture_art:
            return 10000
        else:
            return -100

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.curr_episode += 1
        self.action_episode_memory.append([])
        self.is_picture_art = False
        self.price = 1.00
        return self._get_state()

    def render(self, mode='human', close=False):
        return

    def _get_state(self):
        """Get the observation."""
        ob = [self.TOTAL_TIME_STEPS - self.curr_step]
        return ob
