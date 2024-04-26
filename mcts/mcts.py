import random
from diffusion_policy.env.pusht.pushobjects_rel_env import PushObjectsRelativeEnv
import numpy as np
import torch
from diffusion_policy.common.pytorch_util import dict_apply

class TreeNode:
    def __init__(self, env, policy, parent):
        self.env = env
        self.state = env.state
        self.unexplored_choices = env.get_blocks_todo()
        self.children = []
        self.parent = parent
        self.policy = policy
        self.n_visited = 0
        self.reward = 0
        self.reward_lb = 1

    def select(self):
        # reset state
        self.env._set_state(self.state)
        g_max = 0
        if len(self.unexplored_choices) > 0:
            # we have not fully explored this node yet, so explore
            # choose a node to simulate first
            choice = random.choice(self.unexplored_choices)
            self.unexplored_choices.remove(choice)
            # simulate the choice
            simulate(self.env, choice, self.policy)
            # now make a new child corresponding to that choice
            new_child = TreeNode(self.env, self.policy, self)
            # and play out the policy to see how it does
            reward = new_child.explore()
            self.reward = max(self.reward, reward)
            self.reward_lb = min(self.reward_lb, reward)
            # backpropagate (using max instead of average)
            self.n_visited += 1
            return self.reward
        else:
            # return 1 if we have no options left and no children (terminal state)
            if len(self.children) == 0:
                return 1
            
            # choose child with highest UCT score 
            sel_child = self.best_child()
            reward = sel_child.select()
            self.reward = max(self.reward, reward)
            self.reward_lb = min(self.reward_lb, reward)
            self.n_visited += 1
            return self.reward
        
    def explore(self):
        #choose random actions, we can change this later
        done = len(self.env.get_blocks_todo()) == 0
        iters = 0
        reward = -1
        while not done and iters < 2 * self.env.n_blocks:
            new_idx = random.choice(self.env.get_blocks_todo())
            reward = max(reward, simulate(self.env, new_idx, self.policy))
            done = len(self.env.get_blocks_todo()) == 0
            iters += 1
        self.n_visited += 1
        self.reward = reward
        self.reward_lb = reward
        return reward
    
    def best_child(self):
        #find the child with the highest UCT score
        out = None
        reward = -1
        for child in self.children:
            #from MCTS paper
            q_value = (child.reward - self.reward_lb) / (self.reward - self.reward_lb) + np.sqrt(np.log(self.n_visited)/child.n_visited)
            if reward < q_value:
                reward = q_value
                out = child
        return out
        


def simulate(env: PushObjectsRelativeEnv, idx, policy):
    env.active_idx = idx
    obs = env._get_obs()
    device = policy.device
    past_action = None
    n_action_steps=8
    n_latency_steps=0
    n_obs_steps=8
    policy.reset()

    reward = 0
    done = False
    while not done:
        Do = obs.shape[-1] // 2
        # create obs dict
        np_obs_dict = {
            # handle n_latency_steps by discarding the last n_latency_steps
            'obs': obs[...,:n_obs_steps,:Do].astype(np.float32),
            'obs_mask': obs[...,:n_obs_steps,Do:] > 0.5
        }
        if past_action and (past_action is not None):
            # TODO: not tested
            np_obs_dict['past_action'] = past_action[
                :,-(n_obs_steps-1):].astype(np.float32)
        
        # device transfer
        obs_dict = dict_apply(np_obs_dict, 
            lambda x: torch.from_numpy(x).to(
                device=device))

        # run policy
        with torch.no_grad():
            action_dict = policy.predict_action(obs_dict)

        # device_transfer
        np_action_dict = dict_apply(action_dict,
            lambda x: x.detach().to('cpu').numpy())

        # handle latency_steps, we discard the first n_latency_steps actions
        # to simulate latency
        action = np_action_dict['action'][:,self.n_latency_steps:]

        # step env
        obs, reward, done, info = env.step(action)
        done = np.all(done)
        past_action = action
    return reward

