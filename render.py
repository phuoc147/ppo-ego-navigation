import time
from stable_baselines3 import PPO
from env import CarEnv, WIDTH, HEIGHT
from model import TransformerExtractor  # Needed for loading the model correctly
import pygame as pg
import argparse
# Load environment and model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a trained PPO agent")
    parser.add_argument("--model", type=str, default="ppo_transformer", help="Model name")
    parser.add_argument("--steps", type=int, default=2000, help="Number of steps to run")
    parser.add_argument("--cars_num", type=int, default=50, help="Number of cars in the environment")
    args = parser.parse_args()
    model_name = args.model
    env = CarEnv(args.cars_num,is_training=False,history_len=5,max_collide_num=1)
    model = PPO.load(model_name, env=env)

    screen = pg.display.set_mode((WIDTH, HEIGHT))
    pg.display.set_caption("Simple Game Loop Example")
    # Rendering loop
    obs = env.reset()
    for _ in range(2000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render(screen=screen)
        time.sleep(0.03)  # Slow down for visibility
        if done:
            obs = env.reset()
    env.close()