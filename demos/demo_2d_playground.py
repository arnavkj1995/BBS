"""
A very basic arena for showing off our requirements from the 2D env.
"""
import sys
import numpy as np
import os
from tqdm import tqdm
import argparse
from PIL import Image

from bringbackshapes.twod_playground.arena import Arena
from moviepy.editor import ImageSequenceClip

def main(args):
    arena = Arena(
        render_game=args.render)
    once = False
    obss = []
    pbar = tqdm()
    time_limit = args.time_limit
    step = 0
    while arena.running:
        arena.check_running_status()

        if not args.overide:
            arena.apply_action()
        else:
            arena.apply_action((0, 0))

        arena.update()
        obs = arena.get_observation()
        obss.append(obs)

        if not once:
            obs = np.array(obs)
            once = True

        pbar.update(1)
        step += 1
        if time_limit and step > time_limit:
            break
    
    arena.close()
    pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_limit", type=int, default=None)
    parser.add_argument("--overide", type=int, default=0)
    parser.add_argument("--render", action='store_true')
    args = parser.parse_args()
    sys.exit(main(args))
