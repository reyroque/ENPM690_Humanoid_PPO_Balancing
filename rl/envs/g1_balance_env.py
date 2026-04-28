import gymnasium as gym
import mujoco
import numpy as np


class G1BalanceEnv(gym.Env):
    """Class for the G1 environment."""

    metadata = {"render_modes": [None, "human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None) -> None:
        """Initializer function for this class."""
        super().__init__()

        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        self.render_mode = render_mode
        self.viewer = None

        # Model and data
        self.model = mujoco.MjModel.from_xml_path(
            "third_party/unitree_mujoco/unitree_robots/g1/scene_23dof.xml"
        )
        self.data = mujoco.MjData(self.model)

        # Offscreen renderer

        self.renderer = mujoco.Renderer(self.model)
        # Set default camera after renderer is initialized.
        self.renderer.update_scene(self.data)

        # Spaces
        self.action_dim = self.model.nu
        self.obs_dim = self.model.nq + self.model.nv

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        # Episode configuration
        self.max_steps = 1000
        self.current_step = 0

        self.action_scale = 20.0

        # Human viewer
        if self.render_mode == "human":
            self._launch_viewer()

    def _launch_viewer(self) -> None:
        """Function for the viewer (human mode)."""
        if self.viewer is None:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

            with self.viewer.lock():
                self.viewer.cam.distance = 4.5
                self.viewer.cam.azimuth = 140
                self.viewer.cam.elevation = -15
                self.viewer.cam.lookat[:] = [0.0, 0.0, 0.9]

            self.viewer.sync()

    def reset(self, seed=None, options=None):
        """Function for resetting."""
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0

        # Optional small random perturbation for training robustness
        noise_scale = 0.01
        self.data.qpos[:] += self.np_random.uniform(
            low=-noise_scale, high=noise_scale, size=self.model.nq
        )
        self.data.qvel[:] += self.np_random.uniform(
            low=-noise_scale, high=noise_scale, size=self.model.nv
        )

        mujoco.mj_forward(self.model, self.data)

        if self.render_mode == "human":
            if self.viewer is None or not self.viewer.is_running():
                self._launch_viewer()
            self.viewer.sync()

        return self._get_obs(), {}

    def step(self, action):
        """Function for the step."""
        self.current_step += 1

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        scaled_action = action * self.action_scale
        self.data.ctrl[:] = scaled_action

        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._is_done()
        truncated = self.current_step >= self.max_steps

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        """Function for observation."""
        return np.concatenate(
            [
                self.data.qpos.copy(),
                self.data.qvel.copy(),
            ]
        ).astype(np.float32)

    def _compute_reward(self):
        """Function for computing the reward."""
        torso_height = float(self.data.qpos[2])

        upright_bonus = 1.0
        height_reward = torso_height
        angular_vel_penalty = -0.1 * np.linalg.norm(self.data.qvel[3:6])
        control_penalty = -0.01 * np.linalg.norm(self.data.ctrl)

        return upright_bonus + height_reward + angular_vel_penalty + control_penalty

    def _is_done(self) -> bool:
        """Function to determine if the process is done.

        Returns:
            bool: True if the model has fallen over.
        """
        torso_height = float(self.data.qpos[2])
        return torso_height < 0.4

    def render(self):
        """Function to return a render of the model."""
        # Human mode.
        if self.render_mode == "human":
            if self.viewer is None:
                self._launch_viewer()

            if self.viewer.is_running():
                self.viewer.sync()

            return None

        # RGB mode (needed for OpenCV videos).
        elif self.render_mode == "rgb_array":
            self.renderer.update_scene(self.data)

            frame = self.renderer.render()

            return frame

    def close(self) -> None:
        """Function to close the viewer."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
