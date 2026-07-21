#!/usr/bin/env python3
"""Verify Week 5 dependencies and instantiate all benchmark environments."""

from common import load_config, make_single_agent_env, system_metadata
from train_multi_agent import make_parallel_env, make_sb3_vec_env


def main() -> int:
    config = load_config()
    print("Software:")
    for key, value in system_metadata().items():
        print(f"  {key}: {value}")

    print("\nSingle-agent environments:")
    for env_key, env_cfg in config["single_agent"]["environments"].items():
        env, actual = make_single_agent_env(env_key, env_cfg, seed=7)
        obs, _ = env.reset(seed=7)
        action = env.action_space.sample()
        env.step(action)
        print(
            f"  PASS {env_key} -> {actual}; "
            f"obs_shape={getattr(obs, 'shape', None)} "
            f"action_shape={getattr(action, 'shape', None)}"
        )
        env.close()

    print("\nMulti-agent environment:")
    parallel_env, source = make_parallel_env(config["multi_agent"], seed=7)
    observations, _ = parallel_env.reset(seed=7)
    actions = {
        agent: parallel_env.action_space(agent).sample()
        for agent in parallel_env.agents
    }
    parallel_env.step(actions)
    print(f"  PASS simple_spread_v3 from {source}; agents={len(observations)}")
    parallel_env.close()

    vec_env, _ = make_sb3_vec_env(config["multi_agent"], seed=7)
    vec_env.reset()
    print(f"  PASS SuperSuit -> SB3 VecEnv; num_envs={vec_env.num_envs}")
    vec_env.close()

    print("\nEnvironment check complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
