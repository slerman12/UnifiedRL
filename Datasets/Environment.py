# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import time
from math import inf

from Datasets.Suites import DMC, Atari


class Environment:
    def __init__(self, task_name, frame_stack, action_repeat, max_episode_frames, truncate_episode_frames,
                 seed=0, train=True, suite="DMC", offline=False, batch_size=1, num_workers=1):
        self.suite = suite
        self.offline = offline

        self.env = self.raw_env.make(task_name, frame_stack, action_repeat, max_episode_frames,
                                     truncate_episode_frames, train, seed, batch_size, num_workers)

        self.env.reset()

        self.episode_step = self.last_episode_len = self.episode_reward = self.last_episode_reward = 0
        self.daybreak = None

    @property
    def raw_env(self):
        if self.suite.lower() == "dmc":
            return DMC
        elif self.suite.lower() == "atari":
            return Atari

    def __getattr__(self, item):
        return getattr(self.env, item)

    def rollout(self, agent, steps=inf, vlog=False):
        if self.daybreak is None:
            self.daybreak = time.time()  # "Daybreak" for whole episode

        experiences = []
        vlogs = []

        exp = self.exp

        self.episode_done = False

        if self.offline and agent.training:
            agent.step += 1
            agent.episode += 1
            self.episode_done = True
            return None, None, None

        step = 0
        while not self.episode_done and step < steps:
            # Act
            action = agent.act(exp.observation)
            exp = self.env.step(action.cpu().numpy())

            exp.step = agent.step

            experiences.append(exp)

            if vlog:
                frame = self.env.physics.render(height=256, width=256, camera_id=0) \
                    if hasattr(self.env, 'physics') else self.env.render()
                vlogs.append(frame)

            # Tally reward, done, step
            self.episode_reward += exp.reward.mean()
            self.episode_done = exp.last()
            step += 1

        self.episode_step += step

        if self.episode_done:
            if agent.training:
                agent.episode += 1
            self.env.reset()

            self.last_episode_len = self.episode_step
            self.last_episode_reward = self.episode_reward

        # Log stats
        sundown = time.time()
        frames = self.episode_step * self.action_repeat

        logs = {'time': sundown - agent.birthday,
                'step': agent.step,
                'frame': agent.step * self.action_repeat,
                'episode': agent.episode,
                'reward': self.episode_reward,
                'fps': frames / (sundown - self.daybreak)}

        if self.episode_done:
            self.episode_step = self.episode_reward = 0
            self.daybreak = sundown

        return experiences, logs, vlogs
