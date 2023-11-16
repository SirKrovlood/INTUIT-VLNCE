from habitat_baselines.utils.common import CustomFixedCategorical
from habitat_baselines.rl.ppo.policy import Policy
from torch import nn as nn
import torch
import torch.nn.functional as F
from gym import Space
from habitat import Config

import importlib
from vlnce_baselines.models.encoders.rnn_state_encoder import RNNStateEncoder
#from habitat_baselines.rl.ppo.policy import Net

from  vlnce_baselines.models.cma_policy import CMANet
class BaseIntuitionPolicy(Policy):
    def __init__(self, net, dim_actions, intuition_steps=1):
        super().__init__(net, dim_actions)
        #self.net = net
        #self.dim_actions = dim_actions
        self.intuition_steps = intuition_steps

        delattr(self, "action_distribution")
        print("self.intuition_steps", self.intuition_steps)

        #self.critic = CriticHead(self.net.output_size)

        self.ad_linear = nn.Linear(self.net.output_size,
                                   self.dim_actions*self.intuition_steps)

        nn.init.orthogonal_(self.ad_linear.weight, gain=0.01)
        nn.init.constant_(self.ad_linear.bias, 0)

    def build_distribution(
        self, observations, rnn_hidden_states, prev_actions, masks
    ) -> CustomFixedCategorical:
        #print("observations", observations)
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        #print("features", features.size())
        x = self.ad_linear(features)
        #print("x", x.size())
        #print("x reshaped", torch.reshape(x, (observations['rgb'].size()[0], self.intuition_steps, self.dim_actions)).size())
        #print("x resh all", torch.reshape(x, (observations['rgb'].size()[0], self.intuition_steps, self.dim_actions)))

        distribution = CustomFixedCategorical(
            logits=torch.reshape(x, (observations['rgb'].size()[0], self.intuition_steps, self.dim_actions)))

        return distribution

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        distribution = self.build_distribution(observations, rnn_hidden_states,
                                               prev_actions, masks)

        if deterministic:
            action_seq = distribution.mode()
        else:
            action_seq = distribution.sample()

        action_seq = torch.flatten(action_seq, start_dim=1)

        '''value (1st in return) and action_log_probs (3rd in return)
        are not used in either the train, eval or inference
        so we can skip them
        '''
        return None, action_seq, None, rnn_hidden_states, distribution

class CMAIntuitionPolicy(BaseIntuitionPolicy):
    def __init__(
        self, observation_space: Space, action_space: Space, model_config: Config,
        intuition_steps=1):
        super().__init__(
           CMANet(
                observation_space=observation_space,
                model_config=model_config,
                num_actions=action_space.n,
            ),
            action_space.n,
            intuition_steps
        )

    @classmethod
    def from_config(
        cls, config: Config, observation_space: Space, action_space: Space
    ):
        config.defrost()
        config.MODEL.TORCH_GPU_ID = config.TORCH_GPU_ID
        config.freeze()

        return cls(
            observation_space=observation_space,
            action_space=action_space,
            model_config=config.MODEL,
        )
