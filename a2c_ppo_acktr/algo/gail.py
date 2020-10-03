import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import autograd
import gym
from baselines.common.running_mean_std import RunningMeanStd


class RED(nn.Module):
    def __init__(self, input_dim, hidden_dim, device, sigma, iters):
        super().__init__()
        self.device = device
        self.sigma = sigma
        self.iters = iters

        # This is a random initialization, used to learn
        self.dummytrunk = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            ).to(device)

        self.trunk = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            ).to(device)

        self.trunk.train()
        self.optimizer = torch.optim.Adam(self.trunk.parameters())

    def train_red(self, expert_loader):
        # Train the loader
        self.train()
        for _ in range(self.iters):
            for expert_batch in expert_loader:
                # Get expert state and actions
                expert_state, expert_action = expert_batch
                expert_state = torch.FloatTensor(expert_state).to(self.device)
                expert_action = expert_action.to(self.device)

                # Given expert state and action
                expert_sa = torch.cat([expert_state, expert_action], dim=1)
                fsa = self.trunk(expert_sa)
                with torch.no_grad():
                    fsa_random = self.dummytrunk(expert_sa)

                loss = ((fsa - fsa_random)**2).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print("RED loss: {}".format(loss.data.cpu().numpy()))

    def predict_reward(self, state, action, obfilt=None):
        with torch.no_grad():
            self.eval()
            if obfilt is not None:
                s = obfilt(state.cpu().numpy())
                s = torch.FloatTensor(s).to(action.device)
            else:
                s = state
            d = torch.cat([s, action], dim=1)
            fsa = self.trunk(d)
            fsa_random = self.dummytrunk(d)
            rew = torch.exp(-self.sigma * ((fsa - fsa_random)**2).mean(1))[:, None]
        return rew



class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, device, red=None, sail=False, learn=True):
        super(Discriminator, self).__init__()

        self.device = device

        self.red = red
        self.sail = sail
        self.redtrained = False
        if self.sail:
            assert self.red is not None, 'Cannot run SAIL without using RED'

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)).to(device)

        self.trunk.train()

        self.learn = learn
        self.optimizer = torch.optim.Adam(self.trunk.parameters())

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

    def compute_grad_pen(self,
                         expert_state,
                         expert_action,
                         policy_state,
                         policy_action,
                         lambda_=10):
        alpha = torch.rand(expert_state.size(0), 1)
        expert_data = torch.cat([expert_state, expert_action], dim=1)
        policy_data = torch.cat([policy_state, policy_action], dim=1)

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True

        disc = self.trunk(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def update(self, expert_loader, rollouts, obsfilt=None):
        self.train()
        if obsfilt is None:
            obsfilt = lambda x,y : x

        # If RED is untrained, then train it
        if self.red is not None and not self.redtrained:
            print("Training RED...")
            self.red.train_red(expert_loader) # obsfilt keeps changing after that, Pass the obsfilt to reverse normalized states
            self.redtrained = True
            print("Trained RED.")

        # If there is no SAIL but RED is present,
        # then GAIL doesn't need to be updated
        if self.red is not None and not self.sail:
            return 0

        policy_data_generator = rollouts.feed_forward_generator(
            None, mini_batch_size=expert_loader.batch_size)

        loss = 0
        n = 0
        for expert_batch, policy_batch in zip(expert_loader,
                                              policy_data_generator):
            policy_state, policy_action = policy_batch[0], policy_batch[2]
            policy_d = self.trunk(
                torch.cat([policy_state, policy_action], dim=1))

            expert_state, expert_action = expert_batch
            expert_state = obsfilt(expert_state.numpy(), update=False)
            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)
            expert_d = self.trunk(
                torch.cat([expert_state, expert_action], dim=1))

            expert_loss = F.binary_cross_entropy_with_logits(
                expert_d,
                torch.ones(expert_d.size()).to(self.device))
            policy_loss = F.binary_cross_entropy_with_logits(
                policy_d,
                torch.zeros(policy_d.size()).to(self.device))

            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen(expert_state, expert_action,
                                             policy_state, policy_action)

            loss += (gail_loss + grad_pen).item()
            n += 1

            if self.learn:
                self.optimizer.zero_grad()
                (gail_loss + grad_pen).backward()
                self.optimizer.step()
        return loss / n

    def predict_reward(self, state, action, gamma, masks, update_rms=True, obsfilt=None):
        with torch.no_grad():
            self.eval()
            d = self.trunk(torch.cat([state, action], dim=1))
            s = torch.sigmoid(d)
            # Get RED reward
            if self.red is not None:
                assert self.redtrained
                red_rew = self.red.predict_reward(state, action, obsfilt)

                # Check if SAIL is present or not
                if self.sail:
                    reward = s * red_rew
                else:
                    reward = red_rew
            else:
                # If traditional GAIL
                #reward = s.log() - (1 - s).log()
                reward = - (1 - s).log()

            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CNNDiscriminator(nn.Module):
    def __init__(self, input_shape, action_space, hidden_dim, device, clip=0.01):
        super(CNNDiscriminator, self).__init__()
        self.device = device
        C, H, W = input_shape
        self.n = 0
        if type(action_space) == gym.spaces.box.Box:
            A = action_space.shape[0]
        else:
            A = action_space.n
            self.n = A

        self.main = nn.Sequential(
            nn.Conv2d(C, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU(), Flatten(),
        ).to(device)
        self.clip = clip
        print("Using clip {}".format(self.clip))

        for i in range(4):
            H = (H - 4)//2 + 1
            W = (W - 4)//2 + 1
        # Get image dim
        img_dim = 256*H*W

        self.trunk = nn.Sequential(
            nn.Linear(A + img_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)).to(device)

        self.main.train()
        self.trunk.train()

        self.optimizer = torch.optim.Adam(list(self.main.parameters()) + list(self.trunk.parameters()))
        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

    def compute_grad_pen(self,
                         expert_state,
                         expert_action,
                         policy_state,
                         policy_action,
                         lambda_=10):
        grad_pen = 0
        if True:
            alpha = torch.rand(expert_state.size(0), 1)

            # Change state values
            exp_state = self.main(expert_state)
            pol_state = self.main(policy_state)

            expert_data = torch.cat([exp_state, expert_action], dim=1)
            policy_data = torch.cat([pol_state, policy_action], dim=1)

            alpha = alpha.expand_as(expert_data).to(expert_data.device)

            mixup_data = alpha * expert_data + (1 - alpha) * policy_data

            disc = self.trunk(mixup_data)
            ones = torch.ones(disc.size()).to(disc.device)
            grad = autograd.grad(
                outputs=disc,
                inputs=mixup_data,
                grad_outputs=ones,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]

            grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def update(self, expert_loader, rollouts, obsfilt=None):
        self.train()
        assert obsfilt is None

        policy_data_generator = rollouts.feed_forward_generator(
            None, mini_batch_size=expert_loader.batch_size)

        loss = 0
        n = 0
        for expert_batch, policy_batch in zip(expert_loader,
                                              policy_data_generator):
            policy_state, policy_action = policy_batch[0], policy_batch[2]

            if self.n > 0:
                act = torch.zeros(policy_action.shape[0], self.n)
                polact = policy_action.squeeze()
                act[np.arange(polact.shape[0]), polact] = 1
                policy_action = torch.FloatTensor(act).to(policy_action.device)
                #print('policy', policy_action.shape)

            pol_state = self.main(policy_state)
            policy_d = self.trunk(
                torch.cat([pol_state, policy_action], dim=1))

            expert_state, expert_action = expert_batch
            #print('expert', expert_action.shape)
            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)
            exp_state = self.main(expert_state)
            expert_d = self.trunk(
                torch.cat([exp_state, expert_action], dim=1))

            expert_loss = F.binary_cross_entropy_with_logits(
                expert_d,
                torch.ones(expert_d.size()).to(self.device))
            policy_loss = F.binary_cross_entropy_with_logits(
                policy_d,
                torch.zeros(policy_d.size()).to(self.device))
            #expert_loss = -expert_d.mean().to(self.device)
            #policy_loss = policy_d.mean().to(self.device)

            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen(expert_state, expert_action,
                                             policy_state, policy_action)

            loss += (gail_loss + grad_pen).item()
            n += 1

            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()

            # Clip params here
            #for p in self.parameters():
                #p = p.clamp(-self.clip, self.clip)

        return loss / n

    def predict_reward(self, state, action, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.eval()
            if self.n > 0:
                acts = torch.zeros((action.shape[0], self.n))
                acts[np.arange(action.shape[0]), action.squeeze()] = 1
                acts = torch.FloatTensor(acts).to(action.device)
            else:
                acts = action

            stat = self.main(state)
            d = self.trunk(torch.cat([stat, acts], dim=1))
            s = torch.sigmoid(d)
            reward = -(1 - s).log()
            #reward = d / self.clip
            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)


class ExpertImageDataset(torch.utils.data.Dataset):
    def __init__(self, file_name, train=None, act=None):
        trajs = torch.load(file_name)
        self.observations = trajs['obs']
        self.actions = trajs['actions']
        self.train = train
        self.act = None
        if isinstance(act, gym.spaces.Discrete):
            self.act = act.n

        self.actual_obs = [None for _ in range(len(self.actions))]
        self.lenn = 0
        if train is not None:
            lenn = int(0.8*len(self.actions))
            self.lenn = lenn
            if train:
                self.actions = self.actions[:lenn]
            else:
                self.actions = self.actions[lenn:]

    def __len__(self, ):
        return len(self.actions)

    def __getitem__(self, idx):
        action = self.actions[idx]
        if self.act:
            act = np.zeros((self.act, ))
            act[action[0]] = 1
            action = act
        # Load only the first time, images in uint8 are supposed to be light
        if self.actual_obs[idx] is None:
            if self.train == False:
                image = np.load(self.observations[idx + self.lenn] + '.npy')
            else:
                image = np.load(self.observations[idx] + '.npy')
            self.actual_obs[idx] = image
        else:
            image = self.actual_obs[idx]
        # rescale image and pass it
        img = image / 255.0
        img = img.transpose(2, 0, 1)
        # [C, H, W ] image and [A] actions
        return torch.FloatTensor(img), torch.FloatTensor(action)


class ExpertDataset(torch.utils.data.Dataset):
    def __init__(self, file_name, num_trajectories=4, subsample_frequency=20, train=True, start=0):
        all_trajectories = torch.load(file_name)

        perm = torch.randperm(all_trajectories['states'].size(0))
        #idx = perm[:num_trajectories]
        idx = np.arange(num_trajectories) + start
        if not train:
            assert start > 0

        self.trajectories = {}

        # See https://github.com/pytorch/pytorch/issues/14886
        # .long() for fixing bug in torch v0.4.1
        start_idx = torch.randint(
            0, subsample_frequency, size=(num_trajectories, )).long()

        for k, v in all_trajectories.items():
            data = v[idx]

            if k != 'lengths':
                samples = []
                for i in range(num_trajectories):
                    samples.append(data[i, start_idx[i]::subsample_frequency])
                self.trajectories[k] = torch.stack(samples)
            else:
                self.trajectories[k] = data // subsample_frequency

        self.i2traj_idx = {}
        self.i2i = {}

        self.length = self.trajectories['lengths'].sum().item()

        traj_idx = 0
        i = 0

        self.get_idx = []

        for j in range(self.length):

            while self.trajectories['lengths'][traj_idx].item() <= i:
                i -= self.trajectories['lengths'][traj_idx].item()
                traj_idx += 1

            self.get_idx.append((traj_idx, i))

            i += 1


    def __len__(self):
        return self.length

    def __getitem__(self, i):
        traj_idx, i = self.get_idx[i]

        return self.trajectories['states'][traj_idx][i], self.trajectories[
            'actions'][traj_idx][i]
