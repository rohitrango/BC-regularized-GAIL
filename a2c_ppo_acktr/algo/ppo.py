import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import gym


class PPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 gamma=None,
                 decay=None,
                 act_space=None,
                 ):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.act_space = act_space
        print(self.act_space)

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.gamma = gamma
        self.decay = decay

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def update_bc(self, expert_state, expert_actions, obfilt=None):
        if obfilt:
            expert_state = obfilt(expert_state.cpu().numpy(), update=False)
            expert_state = torch.FloatTensor(expert_state).to(expert_actions.device)
            expert_state = Variable(expert_state)
        if isinstance(self.act_space, gym.spaces.Discrete):
            _expert_actions = torch.argmax(expert_actions, 1)
        else:
            _expert_actions = expert_actions
        values, actions_log_probs, _, _ = self.actor_critic.evaluate_actions(expert_state, None, None, \
                _expert_actions)
        loss = -actions_log_probs.mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                 self.max_grad_norm)
        self.optimizer.step()
        return loss

    def get_action_loss(self, expert_state, expert_actions):
        if isinstance(self.act_space, gym.spaces.Discrete):
            _expert_actions = torch.argmax(expert_actions, 1)
        else:
            _expert_actions = expert_actions
        values, actions_log_probs, _, _ = self.actor_critic.evaluate_actions(expert_state, None, None, \
                _expert_actions)
        loss = -actions_log_probs.mean()
        return loss


    def update(self, rollouts, expert_dataset=None, obfilt=None):
        # Expert dataset in case the BC update is required
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                # Expert dataset
                if expert_dataset:
                    for exp_state, exp_action in expert_dataset:
                        if obfilt:
                            exp_state = obfilt(exp_state.numpy(), update=False)
                            exp_state = torch.FloatTensor(exp_state)
                        exp_state = Variable(exp_state).to(action_loss.device)
                        exp_action = Variable(exp_action).to(action_loss.device)
                        # Get BC loss
                        if isinstance(self.act_space, gym.spaces.Discrete):
                            _exp_action = torch.argmax(exp_action, 1)
                        else:
                            _exp_action = exp_action
                        _, alogprobs, _, _ = self.actor_critic.evaluate_actions(exp_state, None, None, _exp_action)
                        bcloss = -alogprobs.mean()
                        # action loss is weighted sum
                        action_loss = self.gamma * bcloss + (1 - self.gamma) * action_loss
                        # Multiply this coeff with decay factor
                        break

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        if self.gamma is not None:
            self.gamma *= self.decay
            print("gamma {}".format(self.gamma))

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
