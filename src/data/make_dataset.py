import logging
import os
from abc import abstractmethod

import pandas as pd


def example_function():
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    print("This is a print: Importing dataset")
    logger.info(
        "This is a logging with info level: Making final data set from raw data"
    )


class DataLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        if os.path.exists(
            "/".join(dataset_path.split("/")[:-1]) + "/labels/train_labels.csv"
        ):
            self._labels = self._load_labels()

    def load_dataset(
        self,
        patient_id: int,
        max_patient_files_to_load: int = None,
        patient_files_to_skip: int = 0,
        skip_missing_files: bool = False,
    ):
        patient_labels = self._get_patient_labels(patient_id, skip_missing_files)

        if max_patient_files_to_load:
            max_patient_files_to_load = min(
                patient_files_to_skip + max_patient_files_to_load, len(patient_labels)
            )

        dataset = self._load_dataset_given_patient_labels(
            patient_labels[patient_files_to_skip:max_patient_files_to_load]
        )
        return dataset

    def get_generator(self, patient_id: int):
        def generator():
            patient_labels = self._get_patient_labels(
                patient_id, skip_missing_files=True
            )
            for patient_file, label in patient_labels.values:
                yield self._parquet_reader_function(
                    os.path.join(self.dataset_path, patient_file)
                ), label

        return generator()

    def _get_patient_labels(self, patient_id: int, skip_missing_files: bool):
        patient_labels = self._labels[
            self._labels["filepath"].str.contains(f"^{patient_id}")
        ]
        if skip_missing_files:
            # Check if exists the dataset folder or subfolders
            patient_labels = patient_labels[
                patient_labels["filepath"].apply(
                    lambda x: os.path.exists(os.path.join(self.dataset_path, x))
                )
            ]
        return patient_labels

    def _load_dataset_given_patient_labels(self, patient_labels: pd.DataFrame):
        dataset = self._get_empty_dataset()
        for patient_file, label in patient_labels.values:
            dataset = self._create_or_append_patient_file(patient_file, label, dataset)
        return dataset

    def _load_labels(self):
        return pd.read_csv(
            "/".join(self.dataset_path.split("/")[:-1]) + "/labels/train_labels.csv"
        )

    def _create_or_append_patient_file(
        self,
        patient_file: str,
        label: int,
        dataset,
    ):
        current_dataset = self._get_empty_dataset()

        current_file = self._parquet_reader_function(
            os.path.join(self.dataset_path, patient_file)
        )
        current_file["patient_id"] = patient_file
        current_file["label"] = label

        current_dataset = self._concat_datasets([current_dataset, current_file])

        dataset = self._concat_datasets([dataset, current_dataset])
        return dataset

    @staticmethod
    @abstractmethod
    def _get_empty_dataset():
        pass

    @staticmethod
    @abstractmethod
    def _parquet_reader_function(parquet_full_path):
        pass

    @staticmethod
    @abstractmethod
    def _concat_datasets(datasets_list: list):
        pass


class PandasDatasetLoader(DataLoader):
    @staticmethod
    def _concat_datasets(datasets_list):
        return pd.concat(datasets_list)

    @staticmethod
    def _parquet_reader_function(parquet_full_path):
        return pd.read_parquet(parquet_full_path)

    @staticmethod
    def _get_empty_dataset():
        return pd.DataFrame()
