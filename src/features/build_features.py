"""Class that includes methods for the extraction of features from the data.

Typical usage example:

featex = FeatExtractor()
"""

import numpy as np
from scipy import signal
import pandas as pd

class FeatExtractor:

    def __init__(generator):

        self.generator = generator


    def mean(self, data, feats, variable = ""):
        
        if vairable == "":
            feat_means = data.mean()            
        else:
            feat_means = data[variable].mean()

        feat_means.columns = feat_means.columns + "_mean"      

        return pd.concat([feats, feat_means])

    def std(self, data, feats, variable = ""):
        
        if vairable == "":
            feat_stds = data.std()            
        else:
            feat_stds = data[variable].std()

        feat_stds.columns = feat_stds.columns + "_std"      

        return pd.concat([feats, feat_stds])


    

