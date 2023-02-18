import logging
import os
from abc import abstractmethod

import pandas as pd
from sklearn.model_selection import train_test_split


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
    def __init__(self, dataset_path, labels_path: str = "E:/Epilepsy challenge/"):
        self.dataset_path = dataset_path
        self._labels = self._load_labels(labels_path)

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

    def get_generators(
        self,
        patient_id: int,
        feature_extraction_functions: list[tuple[callable, dict]] = [],
    ):
        patient_labels = self._get_patient_labels(patient_id, skip_missing_files=True)
        train_labels, test_labels = self._split_dataset_labels(patient_labels)

        def train_generator():
            for patient_file, label in train_labels.values:
                current_data = self._parquet_reader_function(
                    os.path.join(self.dataset_path, patient_file)
                )
                agg_feat_data = None
                for feature_extraction_function, kwargs in feature_extraction_functions:
                    current_data, agg_feat_data = feature_extraction_function(
                        current_data, agg_feat_data, **kwargs
                    )

                    if current_data == None :
                        break
                    
                    # print(
                    #     f"columns = {agg_feat_data.columns if agg_feat_data is not None else None}"
                    # )
                if agg_feat_data is None:
                    final_data = current_data
                else:
                    final_data = agg_feat_data

                yield final_data, label, patient_file

        def test_generator():
            for patient_file, label in test_labels.values:
                current_data = self._parquet_reader_function(
                    os.path.join(self.dataset_path, patient_file)
                )
                agg_feat_data = None
                for feature_extraction_function, kwargs in feature_extraction_functions:
                    current_data, agg_feat_data = feature_extraction_function(
                        current_data, agg_feat_data, **kwargs
                    )

                if agg_feat_data is None:
                    final_data = current_data
                else:
                    final_data = agg_feat_data

                yield final_data, label, patient_file

        return train_generator(), test_generator()

    def _split_dataset_labels(self, labels: pd.DataFrame):
        labels = self._filter_folders_without_min_files(labels)
        labels = self._prune_labels(labels)
        # print(f"n_labels: {labels}; %_1: {labels['label'].mean()}")

        # Split labels in train and test with sklearn
        train_labels, test_labels = train_test_split(
            labels, test_size=0.5, random_state=42, stratify=labels["label"]
        )
        train_labels, test_labels = train_test_split(
            labels, test_size=0.33, shuffle = False
        ) # Removing the shuffle to keep the events together

        return train_labels, test_labels

    @staticmethod
    def _prune_labels(labels: pd.DataFrame):
        labels["sum_next_5_labels"] = 0
        labels["sum_prev_5_labels"] = 0
        for i in range(7):
            labels["sum_next_5_labels"] += labels["label"].shift(-i)
        for i in range(7):
            labels["sum_prev_5_labels"] += labels["label"].shift(+i)

        mask = (labels["sum_next_5_labels"] > 0) | (labels["sum_prev_5_labels"] > 0)

        final_labels = labels[mask]
        final_labels.drop(columns=["sum_next_5_labels"], inplace=True)
        final_labels.drop(columns=["sum_prev_5_labels"], inplace=True)
        return final_labels

    @staticmethod
    def _filter_folders_without_min_files(labels: pd.DataFrame, min_files: int = 6):
        labels["patient"] = labels["filepath"].apply(lambda x: x[:4])
        labels["folder"] = labels["filepath"].apply(lambda x: x[5:8])
        grouped_labels = labels.groupby(by=["folder"]).count()
        mask = grouped_labels[grouped_labels["filepath"] > min_files]
        labels = labels[labels["folder"].isin(mask.index)]
        labels.drop(columns=["patient", "folder"], inplace=True)
        return labels

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

    @staticmethod
    def _load_labels(labels_path: str):
        return pd.read_csv(os.path.join(labels_path, "train_labels.csv"))

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
