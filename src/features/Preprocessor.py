"""Class that includes methods for the preprocessing the data.

Typical usage example:

preproc = Preprocessor()
"""

import numpy as np
from scipy import signal
import pandas as pd
import math

class Preprocessor:
    """Class that includes methods for the preprocessing the data.
    """
    def __init__(self):
        """init method

        Args:
        """

        self.zi_filter = None
        self.fs = 128 # Hz

    def __validate__(self):
        """ Return a list of valid preprocessor methods to use in train_pipeline function. """    
        return ['remove_nans', 'convert_time_to_radians']
    
    def remove_nans(self, data, feats, variable: str = ""):
        if variable == "":
            num_nans = data.isnull().sum().sum()

            if num_nans > 0:
                data = None
                
        else:
            num_nans = data[variable].isnull().sum()
            nan_columns = variable[num_nans > 0]
            data.drop(columns = nan_columns, inplace = True)
            
        return data, feats
        

    def convert_time_to_radians(self, data, feats):

        dates = pd.to_datetime(data["utc_timestamp"], unit='s')
        hours = dates.dt.hour
        minutes = dates.dt.minute
        seconds = dates.dt.second

        seconds_from_00 = hours*3600 + minutes*60 + seconds
        radians = 2*math.pi*seconds_from_00/(24*3600)

        data["time_sin"] = np.sin(radians)
        data["time_cos"] = np.cos(radians)

        return data, feats

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

    

