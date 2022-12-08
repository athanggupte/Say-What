import os
import logging
import csv
import json
import gymnasium as gym
import time
import datetime
import torch
import numpy as np

import babyai
import babyai.utils as utils
import babyai.rl
from babyai.arguments import ArgumentParser
from babyai.model import ACModel
from babyai.evaluate import batch_evaluate
from babyai.utils.agent import ModelAgent
from minigrid.wrappers import RGBImgPartialObsWrapper

from PIL import Image

ACTIONS = [
    'left',
    'right',
    'forward',
    'pickup',
    'drop',
    'toggle',
    'done',
]

parser = ArgumentParser()
args = parser.parse_args()

env = gym.make(args.env, render_mode="rgb_array")

acmodel = utils.load_model(args.model, raise_not_found=True)

if 'emb' in args.arch:
    obss_preprocessor = utils.IntObssPreprocessor(args.model, env.observation_space, args.pretrained_model)
else:
    obss_preprocessor = utils.ObssPreprocessor(args.model, env.observation_space, args.pretrained_model)

agent = ModelAgent(args.model, obss_preprocessor, argmax=True)
agent.model = acmodel
agent.model.eval()

imgs = []
obs, _ = env.reset()
done = False

while not done:
    action = agent.act(obs)['action']
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    img = env.render()
    print("Action:", ACTIONS[action.item()])
    imgs.append(img)

images = []

for img in imgs:
    im = Image.fromarray(img)
    images.append(im)

images[0].save(args.model + '.gif',
               save_all=True, append_images=images[1:], optimize=False, duration=40, loop=0)