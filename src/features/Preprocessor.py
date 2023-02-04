"""Class that includes methods for the preprocessing the data.

Typical usage example:

preproc = Preprocessor()
"""

import numpy as np
from scipy import signal
import pandas as pd

class Preprocessor:
    """Class that includes methods for the preprocessing the data.
    """
    def __init__(self):
        """init method

        Args:
        """

        self.zi_filter = None
        self.fs = 128 # Hz


    # def remove_baseline(self, sig, order = 1, low_f = 1):

    #     sos = signal.butter(order, low_f, output = 'sos', fs = self.fs)
    #     filtered_sig = signal.sosfilt(sos, sig, self.zi_filter)

    #     return filtered_sig
    
    # def freq_single_filtering(self, sig, freq, order = 1, mode='lowpass'):
    #     """_summary_

    #     Args:
    #         sig
    #         mode (str, optional): _description_. Defaults to "bandpass".
    #         freq

    #     Raises:
    #         ValueError: _description_

    #     Returns:
    #         _type_: _description_
    #     """

    #     sos = signal.butter(order, freq, btype = mode, output = 'sos', fs = self.fs)
    #     filtered_sig = signal.sosfilt(sos, sig, self.zi_filter)
        
    #     else:
    #         raise ValueError("mode is not recognized." + \
    #             f"Expected lowpass or highpass, got {mode}.")

    #     return filtered_sig

    # def freq_band_filtering(self, sig, order = 1, mode='bandpass', low_freq=4,
    #                    high_freq=45):
    #     """_summary_

    #     Args:
    #         sig
    #         order (int, optional)
    #         mode (str, optional): _description_. Defaults to "bandpass".
    #         low_freq (int, optional): _description_. Defaults to 4.
    #         high_freq (int, optional): _description_. Defaults to 45.

    #     Raises:
    #         ValueError: _description_

    #     Returns:
    #         _type_: _description_
    #     """

    #     sos = signal.butter(order, [low_freq, high_freq], btype = mode, output = 'sos', fs = self.fs)
    #     filtered_sig = signal.sosfilt(sos, sig, self.zi_filter)
        
    #     else:
    #         raise ValueError("mode is not recognized." + \
    #             f"Expected bandpass or bandstop, got {mode}.")

    #     return filtered_sig

    def detect_nans(self, df: pd.DataFrame, variable: str = ""):
        if variable == "":
            num_nans = df.isnull().sum().sum()
        else:
            num_nans = df[variable].isnull().sum()
        return num_nans

