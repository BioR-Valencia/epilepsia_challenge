"""Class that includes methods for the extraction of features from the data.

Typical usage example:

featex = FeatExtractor()
"""

import numpy as np
from scipy import signal
import pandas as pd

class FeatExtractor:

    def mean(self, data, feats, variable = ""):
        
        if variable == "":
            feat_means = data.mean(numeric_only = True) 
            feat_means.index = feat_means.index + "_mean"        
        else:
            feat_means = data[variable].mean(numeric_only = True)
            feat_means = pd.Series(feat_means, index = [variable + "_mean"]) 

        return data, pd.concat([feats, feat_means])

    def std(self, data, feats, variable = ""):
        
        if variable == "":
            feat_stds = data.std(numeric_only = True) 
            feat_stds.index = feat_stds.index + "_std"        
        else:
            feat_stds = data[variable].std(numeric_only = True)
            feat_stds = pd.Series(feat_stds, data[variable].columns + "_std") 

        return data, pd.concat([feats, feat_stds])
        


    

