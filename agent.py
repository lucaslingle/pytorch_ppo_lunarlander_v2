import torch as tc


class FullyConnectedAgent(tc.nn.Module):
    def __init__(self, observation_dim, num_features, num_actions):
        super().__init__()
        self.observation_dim = observation_dim
        self.num_features = num_features
        self.num_actions = num_actions

        self.feature_stack = tc.nn.Sequential(
            tc.nn.Linear(self.observation_dim, self.num_features),
            tc.nn.ReLU(),
            tc.nn.Linear(self.num_features, self.num_features),
            tc.nn.ReLU(),
        )
        self.policy_head = tc.nn.Sequential(
            tc.nn.Linear(self.num_features, self.num_actions),
            tc.nn.Softmax(dim=-1)
        )
        self.value_head = tc.nn.Linear(self.num_features, 1)

    def forward(self, x):
        features = self.feature_stack(x)
        policy_probs = self.policy_head(features)
        value_estimate = self.value_head(features)
        return policy_probs, value_estimate