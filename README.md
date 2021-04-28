# pytorch_ppo_lunarlander_v2

### Getting started
Install miniconda; I suggest creating an environment for this project.   
Activate the environment and then install the dependencies:
```
pip3 install torch==1.8.1 torchvision==0.9.1
conda install swig
pip3 install box2d-py
```

### Running
Once everything's installed correctly, you can train the agent via 
```
python main.py --mode=train
```
The agent will consistently attain decent performance with the default settings provided,
landing the lunar lander most of the time in the designated time interval, and never crashing.

It is possible that even better performance can be obtained by hyperparameter tuning, 
but the purpose of this project was to statistically verify that our implementation of the RL algos was sound, 
rather than get the maximum posssible score. 

Once the agent is trained, you can watch it play the game via
```
python main.py --mode=play
```
and a pop-up will appear.
