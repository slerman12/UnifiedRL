![alt text](https://i.imgur.com/0CrFXpN.png)

### Quick Links

- [Setup](#wrench-setting-up)

- [Examples](#mag-sample-scripts)

- [Agents and performances](#bar_chart-agents--performances)

# :runner: Running The Code

To start a train session, once installed:

```
python Run.py
```

Defaults:

```Agent=Agents.DQNAgent```

```task=atari/pong```

Plots, logs, and videos are automatically stored in: ```./Benchmarking```.

![alt text](https://i.imgur.com/2jhOPib.gif)

Welcome ye, weary Traveller.

>Stop here and rest at our local tavern,
>
> Where the most hospitable reinforcements be served, à la carte!

Drink up! :beers:

# :pen: Paper & Citing

For detailed documentation, [see our :scroll:](https://arxiv.com).

```
@inproceedings{yarats2021image,
  title={bla},
  author={Sam Lerman and Chenliang Xu},
  booktitle={bla},
  year={2022},
  url={https://openreview.net}
}
```

If you use any part of this code, **be sure to cite the above!**

An acknowledgment to [Denis Yarats](https://github.com/denisyarats), whose excellent [DrQV2 repo](https://github.com/facebookresearch/drqv2) inspired much of this library and its design.

# :open_umbrella: Unified Learning?
Indeed.

All agents support discrete and continuous control, and offline RL.

See example scripts [below](#mag-sample-scripts).

# :wrench: Setting Up 

Let's get to business.

## 1. Clone The Repo

```
git clone git@github.com:agi-init/UnifiedRL.git
cd UnifiedRL
```

## 2. Gemme Some Dependencies

```
conda env create --name RL --file=Conda.yml
```

## 3. Activate Your Conda Env.

```
conda activate RL
```

Optionally, install Pytorch with CUDA from https://pytorch.org/get-started/locally/.

# :joystick: Installing The Suites 

## 1. Atari Arcade

You can use ```AutoROM``` if you accept the license.

```
pip install autorom
AutoROM --accept-license
```
Then:
```
mkdir ./Datasets/Suites/Atari_ROMS
AutoROM --install-dir ./Datasets/Suites/Atari_ROMS
ale-import-roms ./Datasets/Suites/Atari_ROMS
```
## 2. DeepMind Control
Download MuJoCo from here: https://mujoco.org/download.

Make a ```.mujoco``` folder in your home directory:

```
mkdir ~/.mujoco
```

Extract and move downloaded MuJoCo folder into ```~/.mujoco```. For a linux x86_64 architecture, this looks like:

```
tar -xf mujoco210-linux-x86_64.tar.gz
mv mujoco210/ ~/.mujoco/ 
```

And run:

```
pip install --user dm_control
```

to install DeepMind Control. For any issues, consult the [DMC repo](https://github.com/deepmind/dm_control).

# :file_cabinet: Key files

```Run.py``` handles training and evaluation loops, saving, distributed training, logging, plotting.

```Environment.py``` handles rollouts.

```./Agents``` contains self-contained agents.

# :mag: Sample scripts

### Discrete and continuous action spaces

Humanoid example: 
```
python Run.py task=dmc/humanoid_run
```

DrQV2 Agent in Atari:
```
python Run.py Agent=Agents.DrQV2Agent task=atari/battlezone
```

SPR Agent in DeepMind Control:
```
python Run.py Agent=Agents.SPRAgent task=dmc/humanoid_walk
```

### Offline RL

From a saved experience replay, sans additional rollouts:

```
python Run.py task=atari/breakout offline=true
```

Assumes a replay [is saved](#saving).

### Experiment naming, plotting

The ```experiment=``` flag can help differentiate a distinct experiment; you can optionally control which experiment data is automatically plotted with ```plotting.plot_experiments=```.

```
python Run.py experiment=ExpName1 "plotting.plot_experiments=['ExpName1']"
```

A unique experiment for benchmarking and saving purposes, is distinguished by: ```experiment=```, ```Agent=```, ```task=```, and ```seed=``` flags.

### Saving

Agents can be saved or loaded with the ```save=true``` or ```load=true``` flags.

```
python Run.py save=true load=true
```

An experience replay can be saved or loaded with the ```replay.save=true``` or ```replay.load=true``` flags.

```
python Run.py replay.save=true replay.load=true
```

Agents and replays save to ```./Checkpoints``` and ```./Datasets/ReplayBuffer``` respectively per a unique experiment.

### Distributed

You can share an agent across multiple parallel instances with the ```load_every=``` flag. 

For example, a data-collector agent and an update agent,

```
python Run.py seed_steps=inf replay.save=true load_every=true 
```

```
python Run.py offline=true replay.load=true replay.save=true load_every=true
```

in concurrent processes.

Since both use the same experiment name, they will save and load from the same agent and replay, thereby emulating distributed training.

# :bar_chart: Agents & Performances

# :interrobang: How is this possible

We use our new Creator framework to unify RL discrete and continuous action spaces, as elaborated in our [paper](https://arxiv.com).

Then experience replays are serviceable as datasets for offline RL.

# :mortar_board: Pedagogy and Research

All files are designed to be useful for educational and innovational purposes in their simplicity and structure.

# Note

### If you are interested in the full version of this library, 

Check out our [**UnifiedML**](https://github.com/agi-init/UnifiedML). 

In addition to RL, supports classification and generative modeling, with little overhead.

<hr class="solid">

[MIT License Included.](MIT_LICENSE)