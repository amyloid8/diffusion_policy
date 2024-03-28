from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.pusht_dataset import PushTLowdimDataset
from diffusion_policy.env.pusht.pusht_env import PushTEnv
from diffusion_policy.env.pusht.pymunk_keypoint_manager import PymunkKeypointManager
import numpy as np

def main():
    path = "data/pusht/pusht_cchi_v7_replay.zarr"
    out_path = "data/pusht_rel/pusht_rel_cchi_v7_replay.zarr"
    obs_key='keypoint'
    state_key='state'
    action_key='action'
    print("Running...")
    
    out_buffer = ReplayBuffer.create_from_path(out_path, mode='a')
    
    replay_buffer = ReplayBuffer.copy_from_path(
            path, keys=[obs_key, state_key, action_key])
    goal_pose = np.array([256,256,np.pi/4])

    env = PushTEnv()
    kp_manager = PymunkKeypointManager.create_from_pusht_env(env, rel=True)
    #we only need to calculate keypoints once
    goal_keypoints = kp_manager.get_keypoints_global({"goal": goal_pose})["goal"]
    for episode_idx in range(replay_buffer.n_episodes):
        episode = list()
        for data_idx in range(len(replay_buffer.get_episode(episode_idx)['keypoint'])):
            data = {
                'state': replay_buffer.get_episode(episode_idx)['state'][data_idx],
                'keypoint': np.append(replay_buffer.get_episode(episode_idx)['keypoint'][data_idx], goal_keypoints, axis=0) \
                    - [replay_buffer.get_episode(episode_idx)['state'][data_idx][:2]] * 18,
                'action': np.float32(replay_buffer.get_episode(episode_idx)['action'][data_idx] - replay_buffer.get_episode(episode_idx)['state'][data_idx][:2]),
            }
            episode.append(data)
        data_dict = dict()
        for key in episode[0].keys():
            data_dict[key] = np.stack(
                [x[key] for x in episode])
        out_buffer.add_episode(data_dict, compressors='disk')

    print(out_buffer.n_episodes)
    print(replay_buffer.n_episodes)
    assert(out_buffer.n_episodes == replay_buffer.n_episodes)

    # orig_dataset = PushTLowdimDataset(path)

    # print(len(orig_dataset))
    # print(orig_dataset[0]['obs'].shape)
    

if __name__ == '__main__':
    main()