import argschema

from croissant.schemas import (TrainingException, TrainingSchema)


def train_classifier():
    pass


class ClassifierTrainer(argschema.ArgSchemaParser):
    default_schema = TrainingSchema

    def train(self):
        pass


if __name__ == '__main__':
    trainer = ClassifierTrainer()
    trainer.train()