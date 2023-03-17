"""Class that includes methods for the extraction of features from the data.

Typical usage example:

featex = FeatExtractor()
"""

import numpy as np
from scipy.stats import entropy
import pandas as pd

class FeatExtractor:

    def __validate__(self):
        """ Return a list of valid feature methods to use in train_pipeline function. """    
        return ['coastline', 'hjorth_activity', 'hjorth_complexity', 'hjorth_mobility', 'kurtosis', 
                'mean', 'nonlinear_energy', 'shannon_entropy', 'skewness', 'std']

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
    
    def hjorth_activity(self, data, feats, variable = ""):
        """Hjorth activity.
        https://en.wikipedia.org/wiki/Hjorth_parameters

        Activity: The activity parameter represents the signal power, the variance of a time function.
        """
        if variable == "":
            feat_vars = data.var(numeric_only = True) 
            feat_vars.index = feat_vars.index + "_activity"        
        else:
            feat_vars = data[variable].var(numeric_only = True)
            feat_vars = pd.Series(feat_vars, data[variable].columns + "_activity") 

        return data, pd.concat([feats, feat_vars])

    def hjorth_mobility(self, data, feats, variable = ""):
        """Hjorth mobility.
        https://en.wikipedia.org/wiki/Hjorth_parameters

        Mobility: the ratio of the standard deviation of the first difference of the signal 
        to the standard deviation of the signal.
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

        Complexity: the ratio of the mobility of the first derivate vs the mobility of the signal.
        This is equivalent to compute the ratio between the standard deviation of the first difference ^ 2
        vs the product of the std of the signal and the std of the second difference.
        """

        if variable == "":
            dx = data.diff()
            sx = data.std(numeric_only = True)
            sdx = dx.std(numeric_only = True)
            mobility = sx/sdx

            ddx = dx.diff()
            # dsx = dx.std(numeric_only = True)
            sddx = ddx.std(numeric_only = True)
            dx_mobility = sdx/sddx # dsx/sddx

            complexity = dx_mobility/mobility
            complexity.index = complexity.index + "_complexity"
        else:
            dx = data[variable].diff()
            sx = data[variable].std(numeric_only = True)
            sdx = dx.std(numeric_only = True)
            mobility = sx/sdx

            ddx = dx.diff()
            # dsx = dx.std(numeric_only = True)
            sddx = ddx.std(numeric_only = True)
            dx_mobility = sdx/sddx # dsx/sddx

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

