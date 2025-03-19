import asyncio
import hydra
from omegaconf import DictConfig
from src.vessels_bot import VesselsBot


@hydra.main(version_base="1.3", config_path="./config", config_name="config.yaml")
def main(cfg: DictConfig):
    bot = VesselsBot(
        token=cfg.token,
        save_path=cfg.file_path,
        service_address=cfg.service_address
    )
    asyncio.run(bot.run())


if __name__ == "__main__":
    main()
