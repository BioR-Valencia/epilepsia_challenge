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

    def hjorth_mobility(self, data, feats, variable = ""):
        """Hjorth mobility.
        https://en.wikipedia.org/wiki/Hjorth_parameters
        """
        if variable == "":
            dx = data.diff()
            sx = data.std(numeric_only = True)
            sdx = dx.std(numeric_only = True)
            mobility = sx/sdx
            mobility.index = mobility.index + "_mobility"  
        else:
            dx = data[variable].diff()
            sx = data[variable].std(numeric_only = True)
            sdx = dx.std(numeric_only = True)
            mobility = pd.Series(sdx/sx, data[variable].columns + "_mobility")

        return data, pd.concat([feats, mobility])

    def hjorth_complexity(self, data, feats, variable = ""):
        """Hjorth complexity.
        https://en.wikipedia.org/wiki/Hjorth_parameters
        """
        x = np.insert(data, 0, 0, axis=-1)
        dx = np.diff(x, axis=-1)
        m_dx = self.hjorth_mobility(dx)
        m_x = self.hjorth_mobility(data)
        complexity = np.divide(m_dx, m_x)
        return complexity

    def skewness(self, data, feats, variable = ""):

        if variable == "":
            skew = data.skew(numeric_only = True) 
            skew.index = skew.index + "_skew"        
        else:
            skew = data[variable].std(numeric_only = True)
            skew = pd.Series(skew, data[variable].columns + "_skew") 

        return data, pd.concat([feats, skew])
        
    def kurtosis(self, data, feats, variable = ""):

        if variable == "":
            kurtosis = data.kurtosis(numeric_only = True) 
            kurtosis.index = kurtosis.index + "_kurtosis"        
        else:
            kurtosis = data[variable].std(numeric_only = True)
            kurtosis = pd.Series(kurtosis, data[variable].columns + "_kurtosis") 

        return data, pd.concat([feats, kurtosis])

    def coastline(self, data, feats, variable = ""):

        if variable == "":
            dx = data.diff()
            coastline = dx.sum(numeric_only = True)
            coastline.index = coastline.index + "_coastline"        
        else:
            dx = data[variable].diff()
            coastline = dx.sum(numeric_only = True)
            coastline = pd.Series(coastline, data[variable].columns + "_coastline") 

        return data, pd.concat([feats, coastline])

    

