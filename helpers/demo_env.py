"""
DemoReplayEnv — gymnasium env that replays transitions from a DemoDataset.

On reset(): picks a random episode, returns its first obs_t.
On step():  ignores the input action; returns the demo's obs_next and exposes
            the demo action taken via info["demo_action"] and .current_action.

DemoReplayVecEnv wraps multiple DemoReplayEnv instances (via DummyVecEnv) and
adds get_demo_actions() so DemoReplayPPO can read the per-env demo action before
calling policy.evaluate_actions().
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from demo_dataset import DemoDataset


class DemoReplayEnv(gym.Env):
    """
    Single-env demo replayer.

    Args:
        dataset: DemoDataset instance (normalize=False recommended so obs
                 scale matches the real MetaWorldEnv observations).
    """

    def __init__(self, dataset: DemoDataset):
        obs_dim = dataset.obs_dim
        act_dim = dataset.act_dim

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )

        # Group transitions into per-episode lists
        obs_t    = dataset.obs_t.numpy()
        action   = dataset.action.numpy()
        obs_next = dataset.obs_next.numpy()
        ep_ids   = dataset.ep_ids.numpy()

        self._episodes = []
        for ep_id in np.unique(ep_ids):
            mask = ep_ids == ep_id
            self._episodes.append({
                "obs_t":    obs_t[mask],
                "action":   action[mask],
                "obs_next": obs_next[mask],
            })

        self._ep: dict | None = None
        self._t  = 0
        self._current_action = np.zeros(act_dim, dtype=np.float32)
        self._rng = np.random.default_rng()

    @property
    def current_action(self) -> np.ndarray:
        """Demo action to be taken at the current timestep."""
        return self._current_action

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        ep_idx   = int(self._rng.integers(len(self._episodes)))
        self._ep = self._episodes[ep_idx]
        self._t  = 0
        self._current_action = self._ep["action"][0].copy()
        return self._ep["obs_t"][0].copy(), {}

    def step(self, action):
        ep = self._ep
        t  = self._t

        obs_next    = ep["obs_next"][t].copy()
        demo_action = ep["action"][t].copy()
        done        = (t + 1) >= len(ep["obs_t"])

        rew = -0.1 if not done else 20.0  # small step penalty, reward on episode completion

        self._t += 1
        if not done:
            self._current_action = ep["action"][self._t].copy()

        return obs_next, rew, done, False, {"demo_action": demo_action}

    def render(self):
        pass

    def close(self):
        pass


class DemoReplayVecEnv(DummyVecEnv):
    """
    DummyVecEnv of DemoReplayEnv instances with an extra method to read
    the current demo action across all envs simultaneously.
    """

    def get_demo_actions(self) -> np.ndarray:
        """Returns (n_envs, act_dim) array of the current demo actions."""
        return np.stack([env.current_action for env in self.envs])
