![alt text]()

### Quick Links

- [Setup](#-wrench--setting-up)

- [Examples](#-mag--example-scripts)

- [Agents and performances](#-bar-chart--agents---performance)

# :runner: Running The Code

To start a train session, once installed:

```
python Run.py
```

Defaults:

```Agent=Agents.DQNAgent```

```task=atari/pong```

Plots, logs, and videos are automatically stored in: ```./Benchmarking```.

![alt text]()

Welcome ye, weary Traveller.

>Stop here and rest at our local tavern,
>
> Where all your reinforcements and supervisions be served, à la carte!

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
Yes.

All agents support discrete and continuous control, and offline RL.

See example scripts [below]().

# :wrench: Setting Up 

Let's get to business.

## 1. Clone The Repo

```
git clone github.com/agi-init/UnifiedRL
cd UnifiedRL
```

## 2. Gemme Some Dependencies

```
conda env -f create Conda.yml
```

## 3. Activate Your Conda Env.

```
conda activate RL
```

# :joystick: Installing The Suites 

## 1. Atari Arcade

You can use ```AutoROM``` if you accept the license.

```
pip install autorom
AutoROM --accept-license
```
Then:
```
mkdir Atari_ROMS
AutoROM --install-dir ./Atari_ROMS
ale-import-roms ./ATARI_ROMS
```
## 2. DeepMind Control
Download MuJoCo from here: https://mujoco.org/download.

Make a ```.mujoco``` folder in your home directory:

```
mkdir ~/.mujoco
```

Unrar, unzip, and move (```unrar```, ```unzip```, and ```mv```) downloaded MuJoCo version folder into ```~/.mujoco```. 

And run:
```
pip install git+https://github.com/deepmind/dm_control.git
```

to install DeepMind Control.

Voila.

# :file_cabinet: Key files

```Run.py``` handles training and evaluation loops, saving, distributional training, logging, plotting.

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

Assumes a replay [is saved](saving).

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

[comment]: <> (### Distributional)

[comment]: <> (It is possible to train multiple instances of the same agent with the ```load_every=``` flag. )

[comment]: <> (For example, you can run the following in concurrent processes:)

[comment]: <> (```)

[comment]: <> (python Run.py replay.save=true load_every=true )

[comment]: <> (```)

[comment]: <> (and)

[comment]: <> (```)

[comment]: <> (python Run.py offline=true replay.load=true replay.save=true load_every=true)

[comment]: <> (```)

[comment]: <> (Since both share the same experiment name, they will save and load from the same agent and replay, thereby emulating distributional training.)

# :bar_chart: Agents & Performances

# :interrobang: How is this possible

We use our new Creator framework to unify RL discrete and continuous action spaces, as elaborated in our [paper](https://arxiv.com).

Then treat resulting experience replays as offline RL datasets.

# :mortar_board: Pedagogy and Research

All files are designed to be useful for educational and innovational purposes in their simplicity and structure.

# Note

### If you are interested in the full version of this library, 

Check out our [**UnifiedML**](https://github.com/agi-init/UnifiedML) library. 

In addition to RL, it supports classification, generative modeling, and more.

<hr class="solid">

[MIT License Included.](https://github.com/agi-init/UnifiedRL/MIT_LICENSE)