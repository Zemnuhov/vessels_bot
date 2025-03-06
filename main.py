import asyncio
import hydra
from omegaconf import DictConfig
from src.vessels_bot import VesselsBot


@hydra.main(version_base="1.3", config_path="./config", config_name="config.yaml")
def main(cfg: DictConfig):
    bot = VesselsBot(
        token=cfg.bot_setting.token,
        save_path=cfg.bot_setting.file_path,
        vessels_model=hydra.utils.instantiate(cfg.models.segmentation),
        invasion_model=hydra.utils.instantiate(cfg.models.invasion),
        prediction_device=cfg.bot_setting.prediction_device
    )
    asyncio.run(bot.run())


if __name__ == "__main__":
    main()
