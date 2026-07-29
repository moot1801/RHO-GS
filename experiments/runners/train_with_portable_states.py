"""Train RHO-GS with optimizer/RNG/sampler snapshots at configured iterations."""

from __future__ import annotations

from argparse import ArgumentParser


def main(config_path: str) -> int:
    import Framework
    from Implementations import Datasets as DI
    from Implementations import Methods as MI

    from ..stateful_trainer import StatefulRHOGSTrainer

    Framework.setup(config_path=config_path, require_custom_config=True)
    model = MI.get_model(Framework.config.GLOBAL.METHOD_TYPE, name=Framework.config.TRAINING.MODEL_NAME)
    renderer = MI.get_renderer(Framework.config.GLOBAL.METHOD_TYPE, model=model)
    trainer = StatefulRHOGSTrainer(model=model, renderer=renderer)
    dataset = DI.get_dataset(Framework.config.GLOBAL.DATASET_TYPE, Framework.config.DATASET.PATH)
    trainer.run(dataset)
    Framework.teardown()
    return 0


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    raise SystemExit(main(parser.parse_args().config))
