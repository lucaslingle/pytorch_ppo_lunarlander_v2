import torch as tc
import gym
from agent import FullyConnectedAgent
from runner import Runner
import argparse

parser = argparse.ArgumentParser(description='Trains an agent to land on a moon, using Proximal Policy Optimization')
parser.add_argument('--mode', choices=['train', 'play'], default='play', help='Mode of operation: train or play?')
parser.add_argument('--lr', type=float, default=0.001, help='Adam stepsize parameter.')
parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor.')
parser.add_argument('--gae_lambda', type=float, default=0.95, help='Decay param for Generalized Advantage Estimation.')
parser.add_argument('--ppo_epsilon', type=float, default=0.10, help='Clip param for Proximal Policy Optimization.')
parser.add_argument('--entropy_bonus_coef', type=float, default=0.0, help='Entropy bonus coefficient for PPO.')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='dir name for all checkpoints generated')
parser.add_argument('--model_name', type=str, default='model', help='model name used for checkpoints')
parser.add_argument('--max_steps', type=int, default=int(4*1000*128), help='Total number of environment steps to take.')
args = parser.parse_args()

device = "cuda" if tc.cuda.is_available() else "cpu"
print("Using {} device".format(device))

env = gym.make('LunarLander-v2')
env = gym.wrappers.TransformReward(env, lambda r: r / 20.0)

agent = FullyConnectedAgent(
    observation_dim=8,
    num_features=128,
    num_actions=4
)

optimizer = tc.optim.Adam(agent.parameters(), lr=args.lr)
scheduler = None

runner = Runner(
    env=env,
    gamma=args.gamma,
    gae_lambda=args.gae_lambda,
    ppo_epsilon=args.ppo_epsilon,
    entropy_bonus_coef=args.entropy_bonus_coef,
    checkpoint_dir=args.checkpoint_dir,
    model_name=args.model_name)

runner.maybe_load_checkpoint(agent, optimizer)
if args.mode == 'train':
    runner.train_loop(max_steps=args.max_steps, agent=agent, optimizer=optimizer, scheduler=scheduler, device=device)
elif args.mode == 'play':
    runner.play(max_steps=args.max_steps, agent=agent)