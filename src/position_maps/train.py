import hydra


@hydra.main(config_path="config", config_name="config")
def train(cfg):
    print()


if __name__ == '__main__':
    train()
