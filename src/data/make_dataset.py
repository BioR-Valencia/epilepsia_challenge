import logging

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
