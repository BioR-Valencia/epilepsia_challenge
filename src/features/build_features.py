"""Class that includes methods for the extraction of features from the data.

Typical usage example:

featex = FeatExtractor()
"""

import numpy as np
from scipy.stats import entropy
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

        if variable == "":
            dx = data.diff()
            sx = data.std(numeric_only = True)
            sdx = dx.std(numeric_only = True)
            mobility = sx/sdx

            ddx = dx.diff()
            dsx = dx.std(numeric_only = True)
            sddx = ddx.std(numeric_only = True)
            dx_mobility = dsx/sddx

            complexity = dx_mobility/mobility
            complexity.index = complexity.index + "_complexity"
        else:
            dx = data[variable].diff()
            sx = data[variable].std(numeric_only = True)
            sdx = dx.std(numeric_only = True)
            mobility = sx/sdx

            ddx = dx.diff()
            dsx = dx.std(numeric_only = True)
            sddx = ddx.std(numeric_only = True)
            dx_mobility = dsx/sddx

            complexity = dx_mobility/mobility
            complexity = pd.Series(complexity, data[variable].columns + "_complexity")

        return data, pd.concat([feats, complexity])

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
            dx = data.diff().abs()
            coastline = dx.sum(numeric_only = True)
            coastline.index = coastline.index + "_coastline"        
        else:
            dx = data[variable].diff().abs()
            coastline = dx.sum(numeric_only = True)
            coastline = pd.Series(coastline, data[variable].columns + "_coastline") 

        return data, pd.concat([feats, coastline])

    def nonlinear_energy(self, data, feats):

        data_numerics_only = data.select_dtypes(include=np.number)
        nl_energy = (data_numerics_only.iloc[1:-1]**2 - data_numerics_only.iloc[:-2]*data_numerics_only.iloc[2:]).sum()/data_numerics_only.shape[0]
        nl_energy.index = nl_energy.index + "nl_energy"

        return data, pd.concat([feats, nl_energy])

    def shannon_entropy(self, data, feats):

        data_numerics_only = data.select_dtypes(include=np.number)
        data_numpy = data_numerics_only.to_numpy()
        shannon_entropy = entropy(data_numpy)
        shannon_entropy[shannon_entropy < 0] = 0

        sh_entropy = pd.Series(shannon_entropy, data_numerics_only.columns + "_entropy") 
        return data, pd.concat([feats, sh_entropy])

