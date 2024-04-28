from diffusion_policy.env.pusht.pushobjects_rel_keypoints_env import PushObjectsRelKeypointsEnv
from mcts.mcts import run_mcts

class PushObjectsRKOrch(PushObjectsRelKeypointsEnv):
    def __init__(self,
            legacy=False,
            block_cog=None, 
            damping=None,
            render_size=96,
            keypoint_visible_rate=1.0, 
            agent_keypoints=False,
            draw_keypoints=False,
            reset_to_state=None,
            render_action=True,
            local_keypoint_map=None, 
            color_map=None):
        super().__init__(
            legacy=legacy,
            block_cog=block_cog, 
            damping=damping,
            render_size=render_size,
            keypoint_visible_rate=keypoint_visible_rate, 
            agent_keypoints=agent_keypoints,
            draw_keypoints=draw_keypoints,
            reset_to_state=reset_to_state,
            render_action=render_action,
            local_keypoint_map=local_keypoint_map, 
            color_map=color_map)
        
    
    def step(self, action):
        observation, reward, done, info = super().step(action)
        if done:
            choice = run_mcts(self, 2)
            if choice != self.active_idx:
                self.active_idx = choice
                done = False
        return observation, reward, done, info