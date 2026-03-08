from copy import deepcopy
from typing import Any, ClassVar, TypeVar

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import FloatSchedule, explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3 import PPO

from .buffer import TransitionRolloutBuffer
from .policies import DemoActorCriticPolicy
from helpers.demo_env import DemoReplayVecEnv


from models.vae import VAEInterface, VAEOutput, VAELoss
from models.utils import sc_kl_uniform


from typing import Optional, Tuple, Union, Type, Any

SelfDemoPPO = TypeVar("SelfDemoPPO", bound="DemoPPO")

class DemoPPO(PPO):
    policy: DemoActorCriticPolicy
    rollout_buffer: TransitionRolloutBuffer

    def __init__(
        self,
        policy: DemoActorCriticPolicy,
        env: Union[GymEnv, str],
        null_action: np.ndarray = None,
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 1,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        vae_recon_coef: float = 10.0,
        vae_kl_coef: float = 0.1,
        intrinsic_scale: float = 1.0,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[TransitionRolloutBuffer]] = TransitionRolloutBuffer,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        vae_features_extractor_class: VAEInterface  = None,
        vae_features_extractor_kwargs: Optional[dict[str, Any]] = None,
    ):
        policy_kwargs = policy_kwargs or {}
        policy_kwargs["vae_features_extractor_class"] = vae_features_extractor_class
        policy_kwargs["vae_features_extractor_kwargs"] = vae_features_extractor_kwargs

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
        )

        self.vae_recon_coef = vae_recon_coef
        self.vae_kl_coef = vae_kl_coef

        self.intrinsic_scale = intrinsic_scale

        self.null_action = null_action

        self._prev_last_obs = None
        self._prev_action = None

        if _init_setup_model:
            self._setup_model()

    def set_env(self, env, force_reset: bool = True):
        ret = super().set_env(env, force_reset)
        if force_reset:
            self._prev_last_obs = None
        if force_reset:
            self._prev_action = None
        return ret

    def _setup_learn(self, total_timesteps, callback = None, reset_num_timesteps = True, tb_log_name = "run", progress_bar = False):
        ret = super()._setup_learn(total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar)
        self._prev_last_obs = deepcopy(self._last_obs)
        self._prev_action = np.tile(self.null_action, (self.n_envs, 1))
        return ret

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: TransitionRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        assert self._prev_last_obs is not None, "No previous observation was provided"
        assert self._prev_action is not None, "No previous action was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                s_tm1 = obs_as_tensor(self._prev_last_obs, self.device)  # type: ignore[arg-type]
                a_tm1 = obs_as_tensor(self._prev_action, self.device)  # type: ignore[arg-type]
                s_t = obs_as_tensor(self._last_obs, self.device)  # type: ignore[arg-type]
                if isinstance(env, DemoReplayVecEnv):
                    # Demo mode: use demo actions, evaluate policy on them
                    actions = env.get_demo_actions()  # (n_envs, act_dim) numpy
                    demo_actions_t = obs_as_tensor(actions, self.device)
                    values, log_probs, _ = self.policy.evaluate_actions(s_tm1, a_tm1, s_t, demo_actions_t)
                    clipped_actions = actions
                else:
                    actions, values, log_probs = self.policy.forward(s_tm1, a_tm1, s_t)
                    actions = actions.cpu().numpy()

                    # Rescale and perform action
                    clipped_actions = actions

                    if isinstance(self.action_space, spaces.Box):
                        if self.policy.squash_output:
                            # Unscale the actions to match env bounds
                            # if they were previously squashed (scaled in [-1, 1])
                            clipped_actions = self.policy.unscale_action(clipped_actions)
                        else:
                            # Otherwise, clip the actions to avoid out of bound error
                            # as we are sampling from an unbounded Gaussian distribution
                            clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            # Calculate intrinsic reward using VAE KL per environment
            with th.no_grad():
                s_tp1 = obs_as_tensor(new_obs, self.device)
                a_t = obs_as_tensor(actions, self.device)
                vae_tp1 = self.policy.vae_feature_extractor.forward(s_t, a_t, s_tp1)
                kl_loss = self.policy.vae_feature_extractor.loss(vae_tp1).kl_loss
                intrinsic_rewards = self.intrinsic_scale * kl_loss
                intrinsic_rewards = intrinsic_rewards.view(-1).cpu().numpy()

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(
                            s_t[idx:idx+1], a_t[idx:idx+1], s_tp1[idx:idx+1]
                        ).item()
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                self._prev_last_obs,  # type: ignore[arg-type]
                new_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
                self._prev_action,  # type: ignore[arg-type]
                intrinsic_rewards,
            )
            self._prev_last_obs = self._last_obs  # type: ignore[assignment]
            self._prev_action = actions
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(s_t, a_t, s_tp1)  # type: ignore[arg-type]
            values = values.view(-1).cpu().numpy()

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        vae_losses = []
        vae_recon_losses, vae_kl_losses = [], []

        demo_mode = isinstance(self.env, DemoReplayVecEnv)

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.prev_observations,   # s_{t-1}
                    rollout_data.prev_actions,        # a_{t-1}
                    rollout_data.observations,        # s_t
                    actions,                          # a_t
                )
                values = values.flatten()

                # Calculate intrinsic reward using VAE KL 
                # vae_tp1 = self.policy.vae_feature_extractor.forward(
                #     rollout_data.observations,
                #     rollout_data.actions,
                #     rollout_data.next_observations,
                # )
                # vae_tp1_loss = self.policy.vae_feature_extractor.loss(vae_tp1)
                # intrinsic_rewards = self.intrinsic_scale * vae_tp1_loss.kl_loss.view(-1)

                # values -= intrinsic_rewards.view(-1).detach()
                # values += intrinsic_rewards.view(-1)

                advantages = rollout_data.advantages
                # advantages -= intrinsic_rewards.detach()
                # advantages += intrinsic_rewards
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fractions.append(clip_fraction)
                value_losses.append(value_loss.item())
                approx_kl_divs.append(approx_kl_div)

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                vae_t = self.policy.vae_feature_extractor.forward(
                    rollout_data.prev_observations,
                    rollout_data.prev_actions,
                    rollout_data.observations,
                )

                recon_loss = 0.0
                kl_loss = 0.0

                vae_loss_obj = self.policy.vae_feature_extractor.loss(vae_t)

                recon_loss += vae_loss_obj.recon_loss
                kl_loss += vae_loss_obj.kl_loss.mean()

                # recon_loss += vae_tp1_loss.recon_loss
                # kl_loss += vae_tp1_loss.kl_loss.mean()

                vae_loss = (
                    self.vae_recon_coef * recon_loss
                    + self.vae_kl_coef  * kl_loss
                )

                vae_recon_losses.append(recon_loss.item())
                vae_kl_losses.append(kl_loss.item())
                vae_losses.append(vae_loss.item())
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + vae_loss

                if not demo_mode and self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/vae_loss", np.mean(vae_losses))
        self.logger.record("train/reconstruction_loss", np.mean(vae_recon_losses))
        self.logger.record("train/kl_loss", np.mean(vae_kl_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
