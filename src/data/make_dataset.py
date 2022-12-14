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

    def load_dataset(
        self, max_patients_to_load: int = None, patients_to_skip: int = None
    ):
        patients_list = self._get_patients_list()

        if max_patients_to_load:
            max_patients_to_load = min(
                patients_to_skip + max_patients_to_load, len(patients_list)
            )

        dataset = self._load_dataset_given_patients_list(
            patients_list[patients_to_skip:max_patients_to_load]
        )
        return dataset

    def _get_patients_list(self):
        return sorted(os.listdir(self.dataset_path), key=lambda x: int(x))

    def _load_dataset_given_patients_list(self, patients_list: list[str]):
        dataset = self._get_empty_dataset()
        for patient in patients_list:
            dataset = self._create_or_append_patient(patient, dataset)
        return dataset

    def _create_or_append_patient(
        self,
        patient: str,
        dataset,
    ):
        current_dataset = self._get_empty_dataset()
        patient_path = os.path.join(self.dataset_path, patient)

        for parquet_filename in os.listdir(patient_path)[:5]:
            parquet_full_path = os.path.join(patient_path, parquet_filename)

            current_patient = self._parquet_reader_function(parquet_full_path)
            current_patient["patient_id"] = patient

            current_dataset = self._concat_datasets([current_dataset, current_patient])

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
