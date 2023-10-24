from ldm import util
from omegaconf import OmegaConf
from tqdm import tqdm


def main():
    config = OmegaConf.load(
    "/viscam/u/ksarge/zero123/zero123/configs/sd-objaverse-finetune-c_concat-256.yaml")
    config.data.params.dataset_n_shards = 8
    config.data.params.num_workers = 8
    config.data.params.shuffle_buffer_size = 1
    # config.data.params.dataset_n_shards = 1
    # config.data.params.num_workers = 0

    datamodule = util.instantiate_from_config(config.data)

    for batch in tqdm(datamodule.train_dataloader()):
        break


if __name__ == '__main__':
    main()