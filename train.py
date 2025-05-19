from stable_baselines3 import PPO
from env import CarEnv
from model import TransformerExtractor
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PPO agent with transformer features extractor")
    parser.add_argument("--total_timesteps", type=int, default=50000, help="Total timesteps for training")
    parser.add_argument("--pretrained_model", type=str, default=None, help="Environment name")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--saved_file", type=str, default="ppo_transformer_1", help="File to save the model")
    parser.add_argument("--cars_num", type=int, default=50, help="Number of cars in the environment")
    args = parser.parse_args()

    env = CarEnv(args.cars_num,is_training=True,history_len=5,max_collide_num=1)
    policy_kwargs = dict(
        features_extractor_class=TransformerExtractor,
        features_extractor_kwargs=dict(env=env)
    )
    pretrain_model = args.pretrained_model
    total_timesteps = args.total_timesteps
    batch_size = args.batch_size
    saved_file = args.saved_file
    if pretrain_model:
        print(f"Loading pretrained model from {pretrain_model}")
        model = PPO.load(pretrain_model, env=env, batch_size=batch_size)
    else: model = PPO("MlpPolicy", env=env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./logs", batch_size=batch_size)

    all_params = sum(p.numel() for p in model.policy.parameters())
    total_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}, All parameters:{all_params}")
    model.learn(total_timesteps=total_timesteps, tb_log_name="PPO_Transformer")
    model.save(saved_file)