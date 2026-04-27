import numpy as np


def extract_landmarks(env, scenario_name=None):
    """
    Extract landmark positions from env.world.landmarks.
    Returns flat [2 * n_landmarks] float32 array, or zeros(2) if none.
    """
    positions = []
    for landmark in env.world.landmarks:
        positions.extend(landmark.state.p_pos.tolist())
    if len(positions) == 0:
        return np.zeros(2, dtype=np.float32)
    return np.array(positions, dtype=np.float32)


def get_role_encoding(env, num_adversaries):
    """
    One-hot role encoding: adversary agents = 1.0, good agents = 0.0.
    Returns [n_agents] float32 array.
    """
    roles = np.zeros(env.n, dtype=np.float32)
    for i in range(min(env.n, num_adversaries)):
        roles[i] = 1.0
    return roles


def collect_joint_diffusion_data(env, trainers, arglist):
    """
    Roll out the trained policy in a clean environment and save joint
    (state, action, anchor) trajectories for joint denoiser training.

    Must be called with env and trainers already constructed inside a
    tf.reset_default_graph() + U.single_threaded_session() block (the
    caller in train.py handles the TF session setup).

    Saves a .npz file to arglist.joint_diffusion_data_path with:
        states      [N, H, Ds]      joint observation trajectories
        actions     [N, H, Da]      joint action trajectories
        init_obs    [N, Ds]         initial joint observation (anchor A1)
        landmarks   [N, D_land]     landmark positions (anchor A2)
        roles       [N, n_agents]   agent role encoding (anchor A3)
        state_dim   scalar
        action_dim  scalar
        n_agents    scalar
    """
    H = arglist.diffusion_horizon
    max_episode_len = arglist.max_episode_len
    num_episodes = getattr(arglist, "num_collect_episodes", arglist.num_episodes)

    # Derive dims
    state_dim = sum(int(np.prod(env.observation_space[i].shape)) for i in range(env.n))

    from gym.spaces import Box, Discrete
    action_dim = 0
    for i in range(env.n):
        sp = env.action_space[i]
        if isinstance(sp, Box):
            action_dim += int(np.prod(sp.shape))
        elif isinstance(sp, Discrete):
            action_dim += sp.n
        else:
            raise NotImplementedError("Unsupported action space: {}".format(type(sp)))

    num_adversaries = min(env.n, arglist.num_adversaries)

    print("[JointData] state_dim={}, action_dim={}, horizon={}, collecting {} episodes".format(
        state_dim, action_dim, H, num_episodes))

    state_trajs = []
    action_trajs = []
    init_obs_all = []
    landmarks_all = []
    roles_all = []

    for ep in range(num_episodes):
        obs_n = env.reset()

        # Anchors captured once per episode at t=0 (before any corruption)
        init_obs = np.concatenate(obs_n, axis=0).astype(np.float32)   # [Ds]
        landmarks = extract_landmarks(env, arglist.scenario)            # [D_land]
        roles = get_role_encoding(env, num_adversaries)                 # [n_agents]

        ep_states = []
        ep_actions = []

        for t in range(max_episode_len):
            state_vec = np.concatenate(obs_n, axis=0).astype(np.float32)
            action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
            action_vec = np.concatenate(action_n, axis=0).astype(np.float32)

            ep_states.append(state_vec)
            ep_actions.append(action_vec)

            obs_n, _, done_n, _ = env.step(action_n)
            if all(done_n):
                break

        ep_states = np.asarray(ep_states, dtype=np.float32)
        ep_actions = np.asarray(ep_actions, dtype=np.float32)

        if ep_states.shape[0] < H:
            continue  # skip short episodes

        ep_states = ep_states[:H]
        ep_actions = ep_actions[:H]

        state_trajs.append(ep_states)
        action_trajs.append(ep_actions)
        init_obs_all.append(init_obs)
        landmarks_all.append(landmarks)
        roles_all.append(roles)

        if (ep + 1) % 50 == 0:
            print("[JointData] Collected {}/{} episodes".format(ep + 1, num_episodes))

    states = np.stack(state_trajs, axis=0)       # [N, H, Ds]
    actions = np.stack(action_trajs, axis=0)      # [N, H, Da]
    init_obs_arr = np.stack(init_obs_all, axis=0) # [N, Ds]
    landmarks_arr = np.stack(landmarks_all, axis=0)  # [N, D_land]
    roles_arr = np.stack(roles_all, axis=0)       # [N, n_agents]

    print("[JointData] Dataset shapes:")
    print("  states:   {}".format(states.shape))
    print("  actions:  {}".format(actions.shape))
    print("  init_obs: {}".format(init_obs_arr.shape))
    print("  landmarks:{}".format(landmarks_arr.shape))
    print("  roles:    {}".format(roles_arr.shape))

    np.savez(
        arglist.joint_diffusion_data_path,
        states=states,
        actions=actions,
        init_obs=init_obs_arr,
        landmarks=landmarks_arr,
        roles=roles_arr,
        state_dim=state_dim,
        action_dim=action_dim,
        n_agents=env.n,
    )
    print("[JointData] Saved to {}".format(arglist.joint_diffusion_data_path))
