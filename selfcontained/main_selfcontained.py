import numpy as np
import gym
import tkinter as tk

from PIL import Image, ImageTk
import logging.config

from gym import error, spaces, utils, wrappers, logger

NB_SAMPLES = 5
LEN_IMG = 50


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
        self.is_picture_art_train = [1] * LEN_IMG

        self.current_th_aa = np.random.uniform(0, 127, (3, LEN_IMG, LEN_IMG))
        self.current_th_bb = np.random.uniform(127, 255, (3, LEN_IMG, LEN_IMG))

        self.current_ths_aa = np.zeros(
            (LEN_IMG, LEN_IMG, 3, NB_SAMPLES), dtype=np.uint8)
        self.current_ths_bb = np.zeros(
            (LEN_IMG, LEN_IMG, 3, NB_SAMPLES), dtype=np.uint8)

        self.nb_train = 0
        self.total_rew = 0

    def step(self):

        self.curr_step += 1
        if self.train_phase == True:
            self.current_image = cem(self)

        else:
            cem_evaluate(self)
            self.current_image = cem(self)

        self._take_action(self.current_image)
        reward = self._get_reward()
        return reward, self.is_picture_art

    def _take_action(self, action):
        self.action_episode_memory[self.curr_episode].append(action)

        rs = action[:, :, 0]
        bs = action[:, :, 1]
        gs = action[:, :, 2]

        for x in range(LEN_IMG):
            for y in range(LEN_IMG):
                red = rs[x, y]
                blue = bs[x, y]
                green = gs[x, y]
                self.current_painting[x, y] = [red, blue, green]

    def _get_image(self):
        return self.current_painting

    def itisart(self):
        self.is_picture_art = 50000

    def itisnotart(self):
        self.is_picture_art = -500

    def itisart_train(self, n):
        self.is_picture_art_train[n] = 50000

    def itisnotart_train(self, n):
        # print(n)
        self.is_picture_art_train[n] = -500

    def _get_reward(self):
        if self.is_picture_art:
            return 50000
        else:
            return -500

    def reset(self):

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
    def __init__(self, master, env):
        master.title("isitartRL")
        self.env = env
        self.master = master
        self.nb_train = 0
        master.title("isitartRL")
        button_frame = tk.Frame(master)
        button_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.label = tk.Label(master, height=2, width=30)
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

        img = Image.fromarray(self.env.render('rgb_array'))
        self.img_tk = ImageTk.PhotoImage(img)
        self.image_widget = tk.Label(master, image=self.img_tk)

        self.image_widget.pack(fill="both", expand=True)

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
        if self.nb_train == NB_SAMPLES:
            env.train_phase = False
            env.nb_train = 0
            self.nb_train = 0

        else:
            self.nb_train = self.nb_train + 1
            env.nb_train = env.nb_train + 1
            env.train_phase = True

        self.update_gui()

        reward, _done = self.env.step()
        env.total_rew += reward

        img = Image.fromarray(self.env.render('rgb_array'))
        self.img_tk = ImageTk.PhotoImage(img)
        self.image_widget.configure(image=self.img_tk)
        self.image_widget.pack(fill="both", expand=True)


def cem(self):

    img = np.zeros((LEN_IMG, LEN_IMG, 3))
    for x in range(LEN_IMG):
        for y in range(LEN_IMG):
            for z in range(3):
                current_a = self.current_th_aa[z, x, y]+np.random.normal(0, 1)
                current_b = self.current_th_bb[z, x, y]+np.random.normal(0, 1)
                img[x, y, z] = np.random.uniform(current_a, current_b)
                self.current_ths_aa[x, y, z, self.nb_train-1] = current_a
                self.current_ths_bb[x, y, z, self.nb_train-1] = current_b
    return img


def cem_evaluate(self):

    ys = self.is_picture_art_train
    idx = np.argsort(ys)
    elite_idx = idx[3]

    for x in range(LEN_IMG):
        for y in range(LEN_IMG):
            for z in range(3):

                elite_ths = self.current_ths_aa[x, y, z, :elite_idx]
                elite_std = self.current_ths_bb[x, y, z, :elite_idx]
                th_mean = elite_ths.mean(axis=0)
                th_std = elite_std.mean(axis=0)
                self.current_th_aa[z, x, y] = th_mean
                self.current_th_bb[z, x, y] = th_std


if __name__ == "__main__":
    np.random.seed(1)

    num_steps = 1
    gym.envs.registration.register(id="isitartrl-v1",
                                   entry_point=isitartrlEnv)

    env = gym.make("isitartrl-v1")
    env.reset()

    r = tk.Tk()
    gui = IsItArtRLGUI(r, env)
    r.attributes('-topmost', True)
    r.focus_force()
    r.mainloop()
