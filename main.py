#!/usr/bin/env python
"""This script """

import os
import argparse
from typing import List, Tuple
import pandas as pd
import tensorflow as tf
from datasets.metadata import DatasetMetadata, dataset_metadata
from predictive_models.baseline import Baseline
from predictive_models.window_generator import WindowGenerator


def parse_arguments() -> argparse.Namespace:
    """CLI argument parser"""
    parser = argparse.ArgumentParser(
        description="Time Series Predictive Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", type=str, default="sphere_decay_CFD", help="Dataset name",
    )
    return parser.parse_args()


def read_column_name(filename: str) -> List[str]:
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines


def load_datasets(
    metadata: DatasetMetadata, standardize: bool
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train, validation and test datasets specified by metadata. 
    All data is standardize if standardize is true."""
    train_set = pd.read_csv(
        os.path.join(metadata.dir_path, metadata.train_name), header=0
    )
    validation_set = pd.read_csv(
        os.path.join(metadata.dir_path, metadata.validation_name), header=0,
    )
    test_set = pd.read_csv(
        os.path.join(metadata.dir_path, metadata.test_name), header=0
    )
    columns = read_column_name(os.path.join(metadata.dir_path, metadata.columns_name))
    train_set.columns = columns
    validation_set.columns = columns
    test_set.columns = columns

    if standardize:
        train_set = (train_set - train_set.mean()) / train_set.std()
        validation_set = (validation_set - validation_set.mean()) / validation_set.std()
        test_set = (test_set - test_set.mean()) / test_set.std()

    return train_set, validation_set, test_set


def main():
    """Using a main function to avoid global variables"""
    arguments = parse_arguments()
    train_set, validation_set, test_set = load_datasets(
        dataset_metadata[arguments.dataset], standardize=True
    )
    window_generator = WindowGenerator(
        input_width=32,
        label_width=1,
        shift=1,
        train_df=train_set,
        val_df=validation_set,
        test_df=test_set,
        label_columns=["pressure-force_x"],
    )
    column_indices = {name: i for i, name in enumerate(train_set.columns)}
    validation_performance = dict()
    test_performance = dict()
    models = {"baseline": Baseline}
    for model_name, model_class in models.items():
        model = model_class(label_index=column_indices["pressure-force_x"])
        model.compile(
            loss=tf.losses.MeanSquaredError(), metrics=[tf.metrics.MeanAbsoluteError()]
        )
        validation_performance[model_name] = model.evaluate(window_generator.val)
        test_performance[model_name] = model.evaluate(window_generator.test, verbose=0)


if __name__ == "__main__":
    main()
