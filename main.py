import time
import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import robosuite as suite
from robosuite.wrappers import GymWrapper
from networks import CriticNetwork, ActorNetwork
from buffer import ReplayBuffer
from td3_torch import Agent
import os

# Create the directory if it does not exist
os.makedirs("tmp/td3", exist_ok=True)


if __name__ == "__main__":
    import numpy as np


    def flatten_obs(obs):
        """
        Proper flatten function for robosuite GymWrapper.
        Always flattens in a deterministic robosuite key order.
        """
        if isinstance(obs, dict):
            flat = []
            # robosuite recommended key order
            ordered_keys = [
                "robot0_proprio-state",
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
                "robot0_gripper_qvel",
                "object-state"
            ]
            for k in ordered_keys:
                if k in obs:
                    flat.append(np.asarray(obs[k]).reshape(-1))
            return np.concatenate(flat).astype(np.float32)

        # when GymWrapper gives already flat obs
        return np.asarray(obs, dtype=np.float32).reshape(-1)


    # Load OSC position controller
    controller_config = suite.load_controller_config(
        default_controller="OSC_POSITION"
    )

    # Enable joint velocity mode (1.4+ API)
    controller_config["control_mode"] = "joint_velocity"

    # Create environment
    env = suite.make(
        "Door",
        robots=["Panda"],
        controller_configs=controller_config,
        has_renderer=False,
        use_camera_obs=False,
        horizon=300,
        reward_shaping=True,
        control_freq=20,
    )

    env = GymWrapper(env)
    print("Environment created successfully!")

    actor_learning_rate = 0.001
    critic_learning_rate = 0.001
    batch_size = 128
    layer1_size = 256
    layer2_size = 128

    agent = Agent(actor_learning_rate = actor_learning_rate, critic_learning_rate = critic_learning_rate,tau = 0.005 , input_dims = (env.observation_space.shape[0],), env = env, n_actions = env.action_space.shape[0], layer1_size = layer1_size, layer2_size = layer2_size, batch_size = batch_size)

    writer = SummaryWriter('Logs')
    n_games = 10000 ###
    best_score = 0
    episode_identifier = f"0 - actor_learning_rate = {actor_learning_rate} critic_learning_rate = {critic_learning_rate} layer1_size = {layer1_size} layer2_size = {layer2_size}"

    agent.load_models()
    for  i in range(n_games):
        observation, info = env.reset()
        observation = observation.astype(np.float32)
        input_shape = observation.shape

        done = False
        score = 0

        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_observation = next_observation.astype(np.float32)
            done = truncated or terminated
            score += reward
            agent.remember(observation, action, reward, next_observation, done)
            agent.learn()
            observation = next_observation

        writer.add_scalar(f"Score - {episode_identifier}", score, global_step=i)

        if(i % 10 == 0):
            agent.save_models()

        print(f"Episode:{i} Score:{score}")

