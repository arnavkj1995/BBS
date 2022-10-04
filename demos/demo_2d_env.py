"""
A very basic arena for showing off our requirements from the 2D env.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip

from bringbackshapes.gym_wrappers.twod_playground_env import TwoDPlaygroundEnv


def main(args):
    env = TwoDPlaygroundEnv(
        action_repeat=args.action_repeat,
        dense_reward=True,
        render_game=args.render,
        time_limit=args.time_limit,
        random_start=True,
        max_objects=args.max_objects,
        max_distractors=args.max_distractors,
        variable_num_objects=args.variable_num_objects,
        variable_num_distractors=args.variable_num_distractors,
        variable_goal_position=args.variable_goal_position,
        agent_view_size=args.agent_view_size,
    )
    n_games = args.num_games
    for _ in range(n_games):
        done = False
        tot_r = 0.0
        step = 0
        reww = []
        obs = env.reset()
        obss = []
        pbar = tqdm()

        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            reward != 0.0 and print("Reward: ", reward)
            tot_r += reward
            reww += [reward] * env.action_repeat
            obss.append(obs)
            step += env.action_repeat
            pbar.update(env.action_repeat)

        pbar.close()
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_limit", type=int, default=3000)
    parser.add_argument("--max_objects", type=int, default=5)
    parser.add_argument("--max_distractors", type=int, default=1)
    parser.add_argument("--variable_num_objects", action='store_true')
    parser.add_argument("--variable_num_distractors", action='store_true')
    parser.add_argument("--variable_goal_position", action='store_true')
    parser.add_argument("--agent_view_size", type=int, default=125)
    parser.add_argument("--action_repeat", type=int, default=4)
    parser.add_argument("--num_games", type=int, default=1)
    parser.add_argument("--render", action='store_true')
    args = parser.parse_args()
    sys.exit(main(args))
