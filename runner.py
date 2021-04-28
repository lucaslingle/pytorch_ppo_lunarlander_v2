import numpy as np
import torch as tc
import os

class Runner:
    def __init__(self, env, gamma, gae_lambda, ppo_epsilon, entropy_bonus_coef, model_name, checkpoint_dir):
        # TODO(lucaslingle): add support for td(lambda) based value targets,
        #  standardized advantages, and pop-art normalization for the value function.

        self.env = env
        self.trajectory_steps = 128
        self.trajectories_per_epoch = 16
        self.ppo_opt_iters = 16
        self.ppo_batch_size = 128
        self.environment_steps = 0

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_epsilon = ppo_epsilon
        self.entropy_bonus_coef = entropy_bonus_coef

        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir

    def collect_trajectory(self, agent, initial_observation):
        # this collects tuples of length self.trajectory_steps, starting from the current environment state.
        # it neglects the asymptotic bias induced in the policy gradient by truncating the markov chain,
        # though this effect can be lessened by collecting additional trajectories picking up where we left off.

        # collect tuples of (o_t, a_t, r_t, done_t).
        observations = []
        actions = []
        rewards = []
        dones = []
        initial_observation = initial_observation if initial_observation is not None else self.env.reset()
        observations.append(initial_observation)

        # store policy probabilities pi(a_t|o_t) for actions a_t taken, and value estimates V(o_t).
        policy_probabilities_old = []
        value_estimates_old = []

        agent.eval()
        with tc.no_grad():
            for t in range(1, self.trajectory_steps+1):
                o_t = observations[-1]

                pi_old_t, V_old_t = agent(tc.Tensor(np.expand_dims(o_t, 0)))  # policy dist, value est.
                pi_old_t = pi_old_t.squeeze(0).detach()
                V_old_t = V_old_t.squeeze(0).detach()

                a_t = tc.multinomial(pi_old_t, num_samples=1).squeeze(0)  # sample action from policy dist
                o_tp1, r_t, done_t, _ = self.env.step(a_t.numpy())  # step env
                o_tp1 = self.env.reset() if done_t else o_tp1  # on done transition, next obs should be initial state

                policy_probabilities_old.append(pi_old_t[a_t])
                value_estimates_old.append(V_old_t)

                actions.append(a_t)
                rewards.append(r_t)
                dones.append(done_t)
                observations.append(o_tp1)

            # for the very final observation, we want to append a value estimate for GAE to use,
            # since the trajectory is truncated and may not be done, we will need to bootstrap the advantage estimate
            # from the value function on the final observation.
            if done_t:
                value_estimates_old.append(0.0)
            else:
                _, V_old_T = agent(tc.Tensor(np.expand_dims(o_tp1, 0)))
                V_old_T = V_old_T.squeeze(0).detach()
                value_estimates_old.append(V_old_T)

        return {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "policy_probabilities_old": policy_probabilities_old,
            "value_estimates_old": value_estimates_old
        }

    def collect_experience(self, agent):
        observation_sequences = []
        action_sequences = []
        reward_sequences = []
        done_sequences = []
        policy_probabilities_old_sequences = []
        value_estimates_old_sequences = []

        initial_observation = self.env.reset()
        for i in range(0, self.trajectories_per_epoch):
            print("collecting trajectory {}".format(i))

            trajectory_dict = self.collect_trajectory(agent, initial_observation=initial_observation)

            initial_observation = None if trajectory_dict["dones"][-1] else trajectory_dict["observations"][-1]
            trajectory_dict["observations"] = trajectory_dict["observations"][0:-1]

            observation_sequences.append(trajectory_dict["observations"])
            action_sequences.append(trajectory_dict["actions"])
            reward_sequences.append(trajectory_dict["rewards"])
            done_sequences.append(trajectory_dict["dones"])
            policy_probabilities_old_sequences.append(trajectory_dict["policy_probabilities_old"])
            value_estimates_old_sequences.append(trajectory_dict["value_estimates_old"])

        return {
            "observations": np.array(observation_sequences, dtype=np.float32),
            "actions": np.array(action_sequences, dtype=np.int64),
            "rewards": np.array(reward_sequences, dtype=np.float32),
            "dones": np.array(done_sequences, dtype=bool),
            "policy_probabilities_old": np.array(policy_probabilities_old_sequences, dtype=np.float32),
            "value_estimates_old": np.array(value_estimates_old_sequences, dtype=np.float32)
        }

    def process_experience(self, trajectories):
        """
        Processes a batch of trajectories, apply generalized advantage estimation, while respecting done signals.
        :param trajectories:
        :return: advantage_estimates
        """
        advantage_estimates = np.zeros(dtype=np.float32, shape=(self.trajectories_per_epoch, self.trajectory_steps+1))
        return_estimates = np.zeros(dtype=np.float32, shape=(self.trajectories_per_epoch, self.trajectory_steps+1))
        return_estimates[:,-1] = trajectories['value_estimates_old'][:,-1]

        for t in reversed(range(1, self.trajectory_steps+1)):  # trajectory_steps, ..., 1.
            r_t = trajectories['rewards'][:, t-1]
            V_t = trajectories['value_estimates_old'][:, t-1]
            V_tp1 = trajectories['value_estimates_old'][:, t]
            delta_t = -V_t + r_t + self.gamma * V_tp1

            done_t = trajectories['dones'][:, t-1]
            nonterminal_t = 1.0 - done_t.astype(np.int32)
            A_t = delta_t + (self.gamma * self.gae_lambda) * nonterminal_t * advantage_estimates[:, t]
            R_t = r_t + self.gamma * nonterminal_t * return_estimates[:, t]

            advantage_estimates[:, t-1] = A_t
            return_estimates[:, t-1] = R_t

        return {
            "advantage_estimates": advantage_estimates[:, 0:-1],
            "return_estimates": return_estimates[:, 0:-1]
        }

    def train_loop(self, max_steps, agent, optimizer, scheduler, device):
        # implements PPO algorithm loop as described in Schulman et al., 2017.
        # note that for now our implementation does not use clipping for the value function, as done in baselines/ppo2,
        # as this may require some sort of scaling of the returns in order to work well.

        while self.environment_steps < max_steps:
            trajectories = self.collect_experience(agent)
            processed = self.process_experience(trajectories)

            observations = trajectories['observations']
            pi_olds = trajectories['policy_probabilities_old']
            actions = trajectories['actions']
            values = trajectories['value_estimates_old'][:, 0:-1]

            advantages = processed['advantage_estimates']
            returns = processed['return_estimates']

            agent.train()
            for ppo_iter_i in range(self.ppo_opt_iters):
                print(ppo_iter_i)
                idxs = np.random.randint(
                    low=0, high=(self.trajectories_per_epoch * self.trajectory_steps), size=(self.ppo_batch_size,))

                batch_obs = tc.Tensor(observations[(idxs // self.trajectory_steps), (idxs % self.trajectory_steps)]).float().to(device)
                batch_pi_old = tc.Tensor(pi_olds[(idxs // self.trajectory_steps), (idxs % self.trajectory_steps)]).float().to(device)
                batch_actions = tc.Tensor(actions[(idxs // self.trajectory_steps), (idxs % self.trajectory_steps)]).long().to(device)
                batch_advantages = tc.Tensor(advantages[(idxs // self.trajectory_steps), (idxs % self.trajectory_steps)]).float().to(device)
                batch_values = tc.Tensor(values[(idxs // self.trajectory_steps), (idxs % self.trajectory_steps)]).float().to(device)
                batch_returns = tc.Tensor(returns[(idxs // self.trajectory_steps), (idxs % self.trajectory_steps)]).float().to(device)

                pi_new_vec, V_new = agent(tc.Tensor(batch_obs))
                pi_new = tc.gather(pi_new_vec, dim=-1, index=batch_actions.unsqueeze(-1)).squeeze(-1)

                policy_ratio = pi_new / batch_pi_old
                clipped_policy_ratio = tc.clip(policy_ratio, 1.0-self.ppo_epsilon, 1.0+self.ppo_epsilon)
                ppo_surrogate_objective = tc.mean(
                    tc.min(policy_ratio * batch_advantages, clipped_policy_ratio * batch_advantages)
                )
                ppo_policy_loss = -ppo_surrogate_objective

                vf_mse = tc.mean(
                    tc.square(batch_returns - V_new)
                )
                ppo_value_loss = 0.5 * vf_mse

                entropy = tc.mean(
                    -tc.sum(pi_new_vec * tc.log(pi_new_vec), dim=-1)
                )
                entropy_bonus = self.entropy_bonus_coef * entropy

                composite_loss = ppo_policy_loss + ppo_value_loss - entropy_bonus
                # ^ maximize ppo surrogate objective, minimize value prediction error, and maximize entropy bonus

                optimizer.zero_grad()
                composite_loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()

            self.environment_steps += self.trajectories_per_epoch * self.trajectory_steps
            print("Total environment steps trained on: [{}/{}]".format(self.environment_steps, max_steps))

            self.save_checkpoint(agent, optimizer)

    def play(self, max_steps, agent):
        o_t = self.env.reset()
        total_reward = 0.0

        for t in range(0, max_steps):
            self.env.render()
            pi_t, _ = agent(tc.Tensor(np.expand_dims(o_t, 0)))
            pi_t = pi_t.squeeze(0)
            a_t = tc.multinomial(pi_t, num_samples=1).squeeze(0)
            o_tp1, r_t, done_t, _ = self.env.step(a_t.numpy())
            if done_t:
                print("Episode finished after {} timesteps".format(t + 1))
                break
            o_t = o_tp1
            total_reward += r_t

        print("Total reward: {}".format(total_reward))
        self.env.close()

    def save_checkpoint(self, model, optimizer):
        model_path = os.path.join(self.checkpoint_dir, self.model_name)
        os.makedirs(model_path, exist_ok=True)

        tc.save(model.state_dict(), os.path.join(self.checkpoint_dir, self.model_name, 'model.pth'))
        tc.save(optimizer.state_dict(), os.path.join(self.checkpoint_dir, self.model_name, 'optimizer.pth'))

    def maybe_load_checkpoint(self, model, optimizer):
        try:
            model.load_state_dict(tc.load(os.path.join(self.checkpoint_dir, self.model_name, 'model.pth')))
            optimizer.load_state_dict(tc.load(os.path.join(self.checkpoint_dir, self.model_name, 'optimizer.pth')))
            print('Successfully loaded checkpoint.')
        except Exception:
            print('Bad checkpoint or none. Continuing training from scratch.')
