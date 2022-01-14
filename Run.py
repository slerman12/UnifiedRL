# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import hydra
from hydra.utils import instantiate

import Utils

import torch
torch.backends.cudnn.benchmark = True


# Hydra conveniently and cleanly manages sys args
# Hyper-param cfg files located in ./Hyperparams

@hydra.main(config_path='Hyperparams', config_name='cfg')
def main(args):
    # Set seeds
    Utils.set_seed_everywhere(args.seed)

    args.device = 'gpu' if torch.cuda.is_available() else 'cpu'

    # Train, test environments
    env = instantiate(args.environment)
    generalize = instantiate(args.environment, train=False, seed=args.seed + 1234)

    for arg in ('obs_shape', 'action_shape', 'discrete', 'obs_spec', 'action_spec'):
        setattr(args, arg, getattr(env, arg))

    # Agent
    agent = instantiate(args.agent).to(args.device)

    if args.load:
        Utils.load(args.save_path, agent)

    # Experience replay
    replay = instantiate(args.replay)

    # Loggers
    logger = instantiate(args.logger)

    vlogger = instantiate(args.vlogger)

    # Start
    converged = training = False
    while True:
        # Evaluate
        if agent.step % args.evaluate_per_steps == 0:

            for ep in range(args.evaluate_episodes):
                _, logs, vlogs = generalize.rollout(agent.eval(),  # agent.eval() just sets agent.training to False
                                                    vlog=args.log_video)

                logger.log(logs, 'Eval')

            logger.dump_logs('Eval')

            if args.log_video:
                vlogger.dump_vlogs(vlogs, f'{agent.step}.mp4')

        if agent.step % args.plot_per_steps == 0:
            instantiate(args.plotting)

        if converged:
            break

        # Rollout
        experiences, logs, _ = env.rollout(agent.train(), steps=1)  # agent.train() just sets agent.training to True

        replay.add(experiences)

        if env.episode_done:
            if agent.episode % args.log_per_episodes == 0:
                name = 'Train' if training else 'Seed'
                logger.log(logs, name, dump=True)

            if env.last_episode_len > args.nstep:
                replay.add(store=True)  # Only store full episodes

        converged = agent.step >= args.train_steps
        training = agent.step > args.seed_steps

        if args.load_every:
            Utils.load(args.save_path, agent)

        # Train agent
        if training and agent.step % args.update_per_steps == 0 or converged:

            for _ in range(args.post_updates if converged else 1):  # Additional updates after all rollouts
                logs = agent.learn(replay)  # Trains the agent
                if agent.episode % args.log_per_episodes == 0:
                    logger.log(logs, 'Train')

            if args.save:
                Utils.save(args.save_path, agent)


if __name__ == '__main__':
    main()
