import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from agent.deep_qlearning_agent.deep_qlearning_module import DeepQLearningConfig, DeepQModuleLightning
from src.environment import CartPole

LOG_WANDB = True
USE_GPU = True

if __name__ == '__main__':
    pl.seed_everything(42069)
    environment = CartPole()
    config = DeepQLearningConfig(
        eps_init=1.0,
        eps_final=0.1,
        eps_decay_timesteps=80_000,
        gamma=.99,
        replay_buffer_size=10_000,
        batch_size=64,
        init_buffer_steps=1000,
        lr=5e-4,
        max_score=1000,
    )

    deepq_model = DeepQModuleLightning(
        env=environment,
        config=config,
        device='cpu',  # actually this does nothing lmfao
        optimizer_fn=torch.optim.AdamW,
        log_wandb=LOG_WANDB,
    )

    if LOG_WANDB:
        wandb_logger = WandbLogger(project='deep-qlearning')
        wandb_logger.watch(deepq_model)

    # noinspection PyUnboundLocalVariable
    trainer = pl.Trainer(
        accelerator='cpu' if not USE_GPU or not torch.cuda.is_available() else 'gpu',
        max_epochs=100_000,
        logger=None if not LOG_WANDB else wandb_logger,
        gradient_clip_val=5.0, gradient_clip_algorithm='value',
    )

    trainer.fit(deepq_model)
