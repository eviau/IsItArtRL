import gym
import gym_isitartrl


env = gym.make('isitartrl-v0')

env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())