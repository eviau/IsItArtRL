from __future__ import print_function

import gym
from six.moves import cPickle as pickle
import json
import sys
import os
from os import path
# Different file so it can be unpickled
import argparse

import tkinter as tk

from PIL import Image, ImageTk
import logging.config

from gym import error, spaces, utils, wrappers, logger
from gym.utils import seeding

import numpy as np
import random
import datetime
from numpy import matlib

NB_SAMPLES = 5
LEN_IMG = 250


class MultiDiscrete(gym.Space):
    def __init__(self, nvec):
        """
        nvec: vector of counts of each categorical variable
        """
        self.nvec = np.asarray(nvec, dtype=np.int32)
        gym.Space.__init__(self, self.nvec.shape, np.int32)

    def sample(self):
        return (gym.spaces.np_random.random_sample(self.nvec.shape) * self.nvec).astype(self.dtype)

    def contains(self, x):
        return (0 <= x).all() and (x < self.nvec).all() and x.dtype.kind in 'ui'

    def to_jsonable(self, sample_n):
        return [sample.tolist() for sample in sample_n]

    def from_jsonable(self, sample_n):
        return np.array(sample_n)

    def __repr__(self):
        return "MultiDiscrete({})".format(self.nvec)

    def __eq__(self, other):
        return np.all(self.nvec == other.nvec)


class isitartrlEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.__version__ = "1"
        logging.info("isitartrlEnv - Version {}".format(self.__version__))
        self.TOTAL_TIME_STEPS = 1000000
        self.curr_step = -1
        self.action_space = MultiDiscrete([LEN_IMG, LEN_IMG, 255, 255, 255])
        self.train_phase = True
        low = np.array([0, 0, 0])
        high = np.array([LEN_IMG, LEN_IMG, 3])

        self.observation_space = spaces.Box(low, high, dtype=np.int32)
        self.curr_episode = -1
        self.action_episode_memory = []

        self.current_painting = np.zeros(
            (LEN_IMG, LEN_IMG, 3), dtype=np.uint8)
        self.is_picture_art = False
        self.is_picture_art_train = [0] * LEN_IMG

        # self.current_th_aa = np.random.uniform(0, 127, (3, LEN_IMG, LEN_IMG))
        self.current_th_aa = np.zeros((3, LEN_IMG, LEN_IMG))
        # self.current_th_ab = np.random.uniform(50,127,(3,LEN_IMG,LEN_IMG))

        # temp = np.empty([3,3,LEN_IMG, LEN_IMG])
        # for x in range(LEN_IMG):
        #     for y in range(LEN_IMG):
        #         temp[:,:,x,y] = np.identity(3)

        # self.current_th_ba = np.random.uniform(127,175,(3,LEN_IMG,LEN_IMG))
        # self.current_th_bb = np.random.uniform(127, 255, (3, LEN_IMG, LEN_IMG))
        self.current_th_bb = np.zeros((3, LEN_IMG, LEN_IMG))
        self.current_ths_aa = np.zeros(
            (LEN_IMG, LEN_IMG, 3, NB_SAMPLES), dtype=np.uint8)
        # self.current_ths_ab = np.zeros((LEN_IMG,LEN_IMG,3,NB_SAMPLES), dtype=np.uint8)

        # self.current_ths_ba = np.zeros((LEN_IMG,LEN_IMG,3,NB_SAMPLES), dtype=np.uint8)
        self.current_ths_bb = np.zeros(
            (LEN_IMG, LEN_IMG, 3, NB_SAMPLES), dtype=np.uint8)

        self.nb_train = 0
        self.total_rew = 0

        # image = Image.fromarray(self.current_painting)
        # image.show()

    def step(self):
        # if self.is_picture_art:
        #     raise RuntimeError("It is art ! Marvelous!")
        self.curr_step += 1
        if self.train_phase == True:
            self.current_image = cem(self)

        else:
            cem_evaluate(self)
            self.current_image = cem(self)

        self._take_action(self.current_image)
        reward = self._get_reward()
        # ob = self._get_state()
        return reward, self.is_picture_art

    def _take_action(self, action):
        self.action_episode_memory[self.curr_episode].append(action)
        # print("TEST")
        # print(action_hashed.shape)

        rs = action[:, :, 0]
        bs = action[:, :, 1]
        gs = action[:, :, 2]
        # print(rs.shape)

        # create current_painting
        for x in range(LEN_IMG):
            for y in range(LEN_IMG):
                red = rs[x, y]
                blue = bs[x, y]
                green = gs[x, y]
                # print(red, blue, green)
                self.current_painting[x, y] = [red, blue, green]

    def _get_image(self):
        # image = Image.fromarray(self.current_painting)
        # image.save("isitartRL_" + str(datetime.datetime.now()) + ".png", "PNG")
        return self.current_painting

    def itisart(self):
        self.is_picture_art = True

    def itisnotart(self):
        self.is_picture_art = False

    def itisart_train(self, n):
        # print(self.is_picture_art_train[n])
        self.is_picture_art_train[n] = 1

    def itisnotart_train(self, n):
        # print(n)
        self.is_picture_art_train[n] = 0

    def _get_reward(self):
        if self.is_picture_art:
            return 5000
        else:
            return -500

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
        return self._get_state()

    def render(self, mode='human', close=False):
        if mode == 'rgb_array':
            return self.current_painting

    def _get_state(self):
         """Get the observation."""
         return self.current_painting


class IsItArtRLGUI:
    def __init__(self, master, env, params):
        master.title("isitartRL")
        self.params = params
        self.env = env
        self.master = master
        self.nb_train = 0
        master.title("isitartRL")
        button_frame = tk.Frame(master)
        button_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.label = tk.Label(master, height=2, width=30)
            # T.configure("0.1", "Training phase: is this art ?\n")
        self.label.configure(text="Is it art ? RL edition.")
        self.label.pack()

        self.yes_button = tk.Button(
            button_frame, text='Oui/Yes', command=self.yes)
        self.no_button = tk.Button(
            button_frame, text='Non/No', command=self.no)

        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        self.yes_button.grid(row=1, column=0, sticky=tk.W+tk.E)
        self.no_button.grid(row=1, column=1, sticky=tk.W+tk.E)

        self.img_frame = tk.Frame(master)
        self.img_frame.pack(fill=tk.X, side=tk.TOP)

        # img = Image.open("image.jpg")
        img = Image.fromarray(self.env.render('rgb_array'))
        self.img_tk = ImageTk.PhotoImage(img)
        self.image_widget = tk.Label(master, image=self.img_tk)

        self.image_widget.pack(fill="both", expand=True)
        # env.render()
        # self.env.step(self.env.action_space.sample())

    def update_gui(self):
        if self.nb_train < NB_SAMPLES:
            self.label.configure(
                text="Training phase: is this art ?")

        else:
            self.label.configure(
                text="Reward phase: is this art ?")

    def yes(self):
        self.env.itisart()
        self.step()



    def no(self):
        self.env.itisnotart()
        self.step()



    def step(self):
        # print(self.env.action_space)
        if self.nb_train == NB_SAMPLES:
            env.train_phase=False
            env.nb_train =0
            self.nb_train=0

        else:
            self.nb_train=self.nb_train + 1
            env.nb_train = env.nb_train +1
            env.train_phase=True

        self.update_gui()

        reward, _done = self.env.step()
        env.total_rew += reward

        img=Image.fromarray(self.env.render('rgb_array'))
        self.img_tk=ImageTk.PhotoImage(img)
        self.image_widget.configure(image=self.img_tk)
        self.image_widget.pack(fill="both", expand=True)



def generate_image(th_mean, initial_std=10):
    th_std=np.identity(th_mean.size) * initial_std
    ths=(np.abs(np.floor((np.random.multivariate_normal(
        ((th_mean*np.ones_like(th_mean.size)*255.0)), th_std, (LEN_IMG, LEN_IMG, 3)))))).astype(int)
    return ths


def cem(self):
    """
    Generic implementation of the cross-entropy method for maximizing a black-box function
    f: a function mapping from vector -> scalar
    th_mean: initial mean over input distribution
    batch_size: number of samples of theta to evaluate per batch
    n_iter: number of batches
    elite_frac: each batch, select this fraction of the top-performing samples
    initial_std: initial standard deviation over parameter vectors
    # """
    # n_elite = int(np.round(batch_size*elite_frac))

        # ths = np.array([th_mean + dth for dth in th_std[None, :]
                        # * np.random.randn(batch_size, th_mean.size)])
        # print((th_mean*np.ones_like(th_mean.size)*50.0))
        # ths = (np.abs(np.floor((np.random.multivariate_normal(
        #     ((th_mean*np.ones_like(th_mean.size)*255.0)), th_std, (LEN_IMG, LEN_IMG, 3)))))).astype(int)

    img = np.zeros((LEN_IMG,LEN_IMG,3))    
    for x in range(LEN_IMG):
        for y in range(LEN_IMG):
            for z in range(3):
                current_a = self.current_th_aa[z,x,y]+np.random.uniform(-5,5)
                current_b = self.current_th_bb[z,x,y]+np.random.uniform(-5,5)
                img[x,y,z]=np.random.uniform(current_a,current_b)
                self.current_ths_aa[x,y,z,self.nb_train-1] = current_a
    # print(current_std)
                # self.current_ths_ab[x,y,z,self.nb_train-1] = self.current_th_b[z,x,y]

                # self.current_ths_ba[x,y,z,self.nb_train-1] = self.current_th_a[z,x,y]
    # print(current_std)
                self.current_ths_bb[x,y,z,self.nb_train-1] = current_b
    return img


def cem_evaluate(self):
    """
    Generic implementation of the cross-entropy method for maximizing a black-box function
    f: a function mapping from vector -> scalar
    th_mean: initial mean over input distribution
    batch_size: number of samples of theta to evaluate per batch
    n_iter: number of batches
    elite_frac: each batch, select this fraction of the top-performing samples
    initial_std: initial standard deviation over parameter vectors
    # """
    ys = self.is_picture_art_train
    idx = np.argsort(ys)
    # print(idx)
    elite_idx = idx[2]

    # elite_ths = ths[elite_inds]
        # th_mean = elite_ths.mean(axis=0)
        # th_std = elite_ths.std(axis=0)
    # print(self.current_ths_std)
    for x in range(LEN_IMG):
        for y in range(LEN_IMG):
            for z in range(3):

                elite_ths = self.current_ths_aa[x,y,z,:elite_idx]
                elite_std = self.current_ths_bb[x,y,z,:elite_idx]
    # print("TEST TESTTEST ")
                # print(elite_ths)
                th_mean = elite_ths.mean(axis=0)
                th_std = elite_std.mean(axis=0)
                self.current_th_aa[z,x,y] = th_mean
                self.current_th_bb[z,x,y] = th_std
                print(th_mean)
                # print(th_std)

    # print(th_mean)
    # print(th_std)

# def do_rollout(env, num_steps, render=True):
#     total_rew=0
#     ob=env.reset()
#     for t in range(num_steps):
#         # a = agent.act(ob)
#         # (ob, reward, _done, _info) = env.step(a)
#         reward=env._get_reward
#         total_rew += reward
#         if render:
#             env.render()
#     return total_rew, t+1


def isitartRL_human():
    # agent = BinaryActionLinearPolicy(theta)

    rew=env._get_reward
    return rew


if __name__ == "__main__":
    logger.set_level(logger.INFO)

    np.random.seed(5)

    params=dict(n_iter=10, batch_size=10, elite_frac=0.2)
    num_steps=1
    gym.envs.registration.register(id="isitartrl-v1",
                                   entry_point=isitartrlEnv)

    env=gym.make("isitartrl-v1")
    env.reset()

    r=tk.Tk()
    gui=IsItArtRLGUI(r, env, params)
    r.attributes('-topmost', True)
    r.focus_force()
    r.mainloop()


# if __name__ == '__main__':
#     logger.set_level(logger.INFO)
#     gym.envs.registration.register(id="isitartrl-v1", entry_point=isitartrlEnv)

#     env = gym.make("isitartrl-v1")

#     env.reset()
#     np.random.seed(0)
#     params = dict(n_iter=10000, batch_size=10, elite_frac=0.2)
#     num_steps = 1

#     # You provide the directory to write to (can be an existing
#     # directory, but can't contain previous monitor results. You can
#     # also dump to a tempdir if you'd like: tempfile.mkdtemp().
#     outdir = '/tmp/cem-agent-results'
#     # env = wrappers.Monitor(env, outdir, force=True)

#     # Prepare snapshotting
#     # ----------------------------------------
#     def writefile(fname, s):
#         with open(path.join(outdir, fname), 'w') as fh:
#             fh.write(s)
#     info = {}
#     info['params'] = params
#     info['argv'] = sys.argv
#     info['env_id'] = env.spec.id
#     # ------------------------------------------

#     def noisy_evaluation(theta):
#         agent = BinaryActionLinearPolicy(theta)
#         rew, T = do_rollout(agent, env, num_steps)
#         return rew

#     # Train the agent, and snapshot each stage
#     for (i, iterdata) in enumerate(
#             cem(noisy_evaluation, np.zeros(env.observation_space.shape[0]+1), **params)):
#         print('Iteration %2i. Episode mean reward: %7.3f' %
#               (i, iterdata['y_mean']))
#         agent = BinaryActionLinearPolicy(iterdata['theta_mean'])
#         if args.display:
#             do_rollout(agent, env, 1, render=True)
#         writefile('agent-%.4i.pkl' % i, str(pickle.dumps(agent, -1)))

#     # Write out the env at the end so we store the parameters of this
#     # environment.
#     writefile('info.json', json.dumps(info))

#     env.close()
