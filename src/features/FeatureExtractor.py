"""Feature extractor combines methods to extract features from the EEG data.

Typical usage example:

extractor = FeatureExtractor()
extractor.list_files()
extractor.extract(file_name,[method1,method2])
"""
import os
import pickle
import glob
from pathlib import Path
from tqdm import tqdm
import numpy as np
import mne
from mne.time_frequency import psd_array_welch, psd_array_multitaper
from sklearn.neighbors import KDTree # CAN CHANGE IT TO scipy.spatial.KDTree to get rid of sklearn
from scipy import stats
from scipy.ndimage import convolve1d
import pywt

from brain_datatools.Augmentor import Augmentor

mne.set_log_level(verbose=False, return_old_level=False, add_frames=None)

class FeatureExtractor:
    """Extraction class for producing X_train and Y_train
    depending on selected methods of extraction.

    Attributes:
        base_len (int): number of points in the baseline
        baseline_len_sec (int): length of the baseline in seconds
        headset (str): type of headset used. Normalizations depends on this parameter. 
        root (Path): root location
        info (mne.Info): mne info to be used in feature extraction methods
        methods (dict): list of feature extraction methods
        norm_dict (dict): list of available normalizations
    """
    
    def __init__(self, root="", experiment="test", headset = None,norm = None, verbose=False):

        # unsupported operand type(s) for /: 'WindowsPath' and 'NoneType'
        if experiment is None:
            experiment=""

        if (root is None) or (root==""):
            home = Path.home()
            self.root = home / ".brainamics" / experiment
        else:
            self.root = Path(root) / experiment

        if verbose:
            print(f"Root experiment folder is {self.root}")
            print("==================================================")

            print(self.list_files())

        self.headset = headset
        if norm is None:
            norm="none"
        self.norm = norm
        loc_dir = os.path.dirname(__file__)
        filename = os.path.join(loc_dir, 'config','headset_characteristics.pickle')
        with open(filename, 'rb') as handle:
            self.headset_dict = pickle.load(handle)

        self.augmentor = Augmentor()
        self.augmentation_list = []

        self.info = None
        self.bands = { "delta": [0, 4],
                       "theta": [4, 8],
                       "alpha": [8, 12],
                       "low_alpha": [8, 10],
                       "high_alpha": [10, 12],
                       "beta": [12,30],
                       "gamma": [32, 60]}

        self.baseline_s = 10
        self.methods = {
            "psd": self.psd,
            "variance": self.variance,
            "hjorth_mobility": self.hjorth_mobility,
            "hjorth_complexity": self.hjorth_complexity,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "sample_entropy": self.sample_entropy,
            "katz_fractal": self.katz_fractal,
            # "wavelet_coef_energy": self.wavelet_coef_energy,  # currently too many features
            "ptp_amp": self.ptp_amp,
            "higuchi_fractal": self.higuchi_fractal,
            "zero_cross": self.zero_cross,
            "spectral_entropy": self.spectral_entropy,
            "teager_kaiser_energy": self.teager_kaiser_energy,  # currently too many features
            "hjorth_mobility_spect": self.hjorth_mobility_spect,
            "hjorth_complexity_spect": self.hjorth_complexity_spect,
            # "phase_lock_val": self.phase_lock_val,  # did not test BIVARIATE methods yet
            #"spect_bands": self.spect_bands, # currently too many features
            "theta_band": self.theta_band,
            "alpha_band": self.alpha_band,
            "low_alpha_band": self.low_alpha_band,
            "high_alpha_band": self.high_alpha_band,
            "beta_band": self.beta_band,
            "gamma_band": self.gamma_band,
            "a_b_ratio": self.a_b_ratio,
            "b_g_ratio": self.b_g_ratio,
            "th_b_ratio": self.th_b_ratio,
        }
        self.norm_dict = {
            "none" : self.dummy_normalize,
            "log" : self.log_normalize_features,
            "boxcox" : self.boxcox_normalize_features,
            "tanh" : self.tanh_normalize_features,
        }

    def list_files(self):
        """Function to print available files to be processed.
        """
        pp_files = glob.glob(str(Path(self.root) / "preprocessed" / "*.pkl"))
        pp_files = [item.replace("/", "\\").split("\\")[-1].split(".")[0] for item in pp_files]
        print("Files ready for feature extraction: ")
        for file_name in pp_files:
            print(file_name)

    def extract(self, folder, methods, is_train = True):
        """Extracts the whole data given in necessary folder. Writes
        the X_test Y_test as a pickle folder.

        Args:
            folder ("string"): Name of the folder to be operated on
            methods (List[str]): Names of the extraction methods

        Raises:
            ValueError: File does not exists, or wrong path
        """
        if not os.path.exists(Path(self.root) / "preprocessed" / (folder + ".pkl")):
            raise ValueError(
                f"{folder} does not exists use list_files " \
                    "function to see ready files to be processed")

        with open(Path(self.root) / "preprocessed" / (folder + ".pkl"), "rb") as file_handl:
            pkl_file = pickle.load(file_handl)

        X = []
        Y = []
        participant_ids = []

        metadata = pkl_file[list(pkl_file.keys())[0]]["metainfo"].copy()
        metadata["features"] = methods
        del metadata["s_freq"]
        metadata["stage"] = "feature_extraction"

        for _, payload in tqdm(pkl_file.items()):
            self.baseline_s = int(payload["metainfo"]["baseline_len_sec"])
            self.info = payload["mne_info"]
            part_id = payload["metainfo"]['participant_id']

            # augmentation starts here
            # produces the shape (#frames, (data.shape))
            data_video = payload["data"][:, int(self.baseline_s * payload["mne_info"]["sfreq"]):]
            data_baseline = payload["data"][:, :int(self.baseline_s * payload["mne_info"]["sfreq"])]
            data_frames = self.augmentor.augment_data(data_video,
                                                      payload["mne_info"].copy(),
                                                      methods=self.augmentation_list)

            Y_tmp = np.array(payload["labels"])
            X_tmb_base = self.extract_features(data_baseline,
                                               methods,
                                               payload["mne_info"])

            for frame_id in range(data_frames.shape[0]):
                X_tmp = self.extract_features(data_frames[frame_id],
                                              methods,
                                              payload["mne_info"])
                if is_train :
                    X_tmp = ((X_tmp - X_tmb_base) / X_tmb_base).flatten()
                else :
                    X_tmp = ((X_tmp - X_tmb_base) / X_tmb_base)

                X.append(X_tmp)
                Y.append(Y_tmp)
                participant_ids.append(part_id)
        X = np.array(X)
        del metadata["participant_id"]
        print("Final shape of X_train:")
        print(X.shape)
        print(f"Feature extraction for Folder: {folder} is complete")
        deliverable = {"X_train": np.array(X),
                       "Y_train": np.array(Y),
                       "metainfo": metadata,
                       "participant_ids": np.array(participant_ids)}

        # saving the file
        fld_name = folder.split("_")[0]
        file_name = Path(self.root) / "extracted" / (fld_name + "_features.pkl")
        print("Saved in: ",file_name)
        if not os.path.exists(file_name.parent):
            (file_name.parent).mkdir(parents=True, exist_ok=True)
        with open(file_name, "wb") as a_file:
            pickle.dump(deliverable, a_file)
        return file_name

    def extract_features(self, data, feat_names, info):
        """Prepare step that is used internally (or externally for testing).
        Prepares one example data entry with given parameters.

        Args:
            data (np.array): Data to be processed (#channels, #points)
            info (mne.info): MNE Info object, holding necessary information of the signal
            feat_names (list[str]): Features to be extracted. Example ["variance", "zero_cross"]

        Raises:
            TypeError: No list of methods given
            TypeError: info is not of type mne.Info
            TypeError: List of methods have to be strings
            ValueError: If method does not exists in dictionary

        Returns:
            np.array: shape = (#electrodes, #features, #points per feature)
        """
        if info:
            # meaning info is hand given
            if not isinstance(info, mne.Info):
                raise TypeError(f"Expected info to be type mne.Info received {type(info)}")
            self.info = info
        if not feat_names:
            raise TypeError("Please enter at least 1 "
                            "method name for preparation...")
        if not isinstance(feat_names, list):
            raise TypeError(f"Expected list of method names "
                            f"in string as input, not {type(feat_names)}")

        x_out_vec = []
        for method in feat_names:
            if method not in self.methods.keys():
                raise ValueError(f"Method: {method} is not available or not found.")
            tmp = self.methods[method](data)
            # check if tmp lies between 5 std from the mean (if statement)
            x_out_vec.append(np.array(tmp)[:, None])
            #print(method, np.array(tmp)[:, None].shape)
        x_out_vec = np.concatenate(x_out_vec,axis=1)
        x_out_vec = self.norm_dict[self.norm](x_out_vec,feat_names)
        return x_out_vec


    def dummy_normalize(self, X, features):
        """Dummy function"""
        return X

    def log_normalize_features(self, X, features):
        """Log Normalization of a feature array

        Args:
            X (np.ndarray): feature array to be normalized (#channels, #features)
            features (list[str]): Features contained in the feature array.
                Example ["variance", "zero_cross"]. (#features)

        Returns :
            np.ndarray : The log normalized feature array (#channels, #features)
        """
        X_log = []
        feature_means = []
        feature_stds = []
        for count,method in enumerate(features):
            mean,std,offset = self.headset_dict["log"][method][self.headset]
            X_log.append(np.log(X[:,count]+offset)[:,None])
            feature_means.append(mean)
            feature_stds.append(std)
        X_log = np.concatenate(X_log,axis=1)
        return (X_log-np.asarray(feature_means)[None,:])/np.asarray(feature_stds)[None,:]

    def tanh_normalize_features(self, X, features):
        """tanh Normalization of a feature array

        Args:
            X (np.ndarray): feature array to be normalized (#channels, #features)
            features (list[str]): Features contained in the feature array.
                Example ["variance", "zero_cross"]. (#features)

        Returns :
            np.ndarray : The tanh normalized feature array (#channels, #features)
        """
        X_tanh = []
        feature_means = []
        feature_stds = []
        for count,method in enumerate(features):
            mean,std,param = self.headset_dict["tanh"][method][self.headset]
            X_tanh.append(np.tanh(X[:,count]/param)[:,None])
            feature_means.append(mean)
            feature_stds.append(std)
        X_tanh = np.concatenate(X_tanh,axis=1)
        return (X_tanh-np.asarray(feature_means)[None,:])/np.asarray(feature_stds)[None,:]

    def boxcox_normalize_features(self, X, features):
        """boxcox Normalization of a feature array

        Args:
            X (np.ndarray): feature array to be normalized (#channels, #features)
            features (list[str]): Features contained in the feature array.
                Example ["variance", "zero_cross"]. (#features)

        Returns:
            np.ndarray: The boxcox normalized feature array (#channels, #features)
        """
        shape = X.shape
        X_boxcox = []
        feature_means = []
        feature_stds = []
        for count,method in enumerate(features):
            mean,std,offset = self.headset_dict["boxcox"][method][self.headset]
            print(X[:,count]+offset)
            X_boxcox.append((X[:,count]+offset)[:,None])
            feature_means.append(mean)
            feature_stds.append(std)
        X_boxcox = np.concatenate(X_boxcox,axis=1)
        X_boxcox = [stats.boxcox(y)[0] for y in np.reshape(X_boxcox,(np.prod(shape[:-1]),-1))]
        X_boxcox = np.reshape(X_boxcox,shape)
        return (X_boxcox-np.asarray(feature_means)[None,:])/np.asarray(feature_stds)[None,:]

    def list_features(self):
        """Used to check available methods for extraction"""
        print("Available methods:")
        for key in self.methods:
            print(key)

    # TIME BASED
    def variance(self, data):
        """Variance with ddof=1

        Args:
            data (np.ndarray): EEG data with the shape of (n_ch, n_points)

        Returns:
            np.ndarray: with the shape (n_ch, n_features per channel)
        """
        return np.var(data, axis=-1, ddof=1)

    def hjorth_mobility(self, data):
        """Hjorth mobility.
        https://en.wikipedia.org/wiki/Hjorth_parameters

        Args:
            data (np.ndarray): EEG data with the shape of (n_ch, n_points)

        Returns:
            np.ndarray: with the shape (n_ch, n_features per channel)
        """
        x = np.insert(data, 0, 0, axis=-1)
        dx = np.diff(x, axis=-1)
        sx = np.std(x, ddof=1, axis=-1)
        sdx = np.std(dx, ddof=1, axis=-1)
        mobility = np.divide(sdx, sx)
        return mobility

    def hjorth_complexity(self, data):
        """Hjorth complexity.
        https://en.wikipedia.org/wiki/Hjorth_parameters

        Args:
            data (np.ndarray): EEG data with the shape of (n_ch, n_points)

        Returns:
            np.ndarray: with the shape (n_ch, n_features per channel)
        """
        x = np.insert(data, 0, 0, axis=-1)
        dx = np.diff(x, axis=-1)
        m_dx = self.hjorth_mobility(dx)
        m_x = self.hjorth_mobility(data)
        complexity = np.divide(m_dx, m_x)
        return complexity

    def skewness(self, data):
        """Skewness

        Args:
            data (np.ndarray): EEG data with the shape of (n_ch, n_points)

        Returns:
            np.ndarray: with the shape (n_ch, n_features per channel)
        """
        return stats.skew(data, axis=1)

    def kurtosis(self, data):
        """Kurtosis

        Args:
            data (np.ndarray): EEG data with the shape of (n_ch, n_points)

        Returns:
            np.ndarray: with the shape (n_ch, n_features per channel)
        """
        return stats.kurtosis(data, axis=1)

    def _embed(self, x, d, tau):
        """Time-delay embedding.
        Parameters

        Args:
            x (ndarray): shape (n_channels, n_times)
            d (int): Embedding dimension.
            tau (int): Delay.
                        The delay parameter ``tau`` should be less or equal than
                        ``floor((n_times - 1) / (d - 1))``.

        Raises:
            Warning: if tau > tau_max = floor((n_times - 1) / (d - 1))

        Returns:
            ndarray: shape (n_channels, n_times - (d - 1) * tau, d)
        """
        tau_max = np.floor((x.shape[1] - 1) / (d - 1))
        if tau > tau_max:
            _tau = tau_max
            raise Warning(f'The given value {tau} for the parameter `tau` exceeds ',
                             f'`tau_max = {tau_max}`. Using `tau`={tau_max} instead.')
        else:
            _tau = int(tau)
        x = x.copy()
        X = np.lib.stride_tricks.as_strided(
            x, (x.shape[0], x.shape[1] - d * _tau + _tau, d),
            (x.strides[-2], x.strides[-1], x.strides[-1] * _tau))
        return X

    def _app_samp_entropy_helper(self, data, emb, metric='chebyshev',
                                approximate=True):
        """Utility function for `compute_app_entropy`` and `compute_samp_entropy`.

        Args:
            data (np.ndarray): EEG data with the shape of (n_ch, n_points)
            emb (int): Embedding dimension.
            metric (str, optional): Name of the metric function used with KDTree. The list of available
                                    metric functions is given by: ``KDTree.valid_metrics``.. Defaults to 'chebyshev'.
            approximate (bool, optional): If True, the returned values will be used to compute the
                                            Approximate Entropy (AppEn). Otherwise, the values are used to compute
                                            the Sample Entropy (SampEn).. Defaults to True.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        _all_metrics = KDTree.valid_metrics
        if metric not in _all_metrics:
            raise ValueError(f'The given metric {metric} is not valid. The valid '
                            f'metric names are: {_all_metrics}')
        n_channels, _ = data.shape
        phi = np.empty((n_channels, 2))
        for j in range(n_channels):
            r = 0.2 * np.std(data[j, :], axis=-1, ddof=1)
            # compute phi(emb, r)
            _emb_data1 = self._embed(data[j, None], emb, 1)[0, :, :]
            if approximate:
                emb_data1 = _emb_data1
            else:
                emb_data1 = _emb_data1[:-1, :]
            count1 = KDTree(emb_data1, metric=metric).query_radius(
                emb_data1, r, count_only=True).astype(np.float64)
            # compute phi(emb + 1, r)
            emb_data2 = self._embed(data[j, None], emb + 1, 1)[0, :, :]
            count2 = KDTree(emb_data2, metric=metric).query_radius(
                emb_data2, r, count_only=True).astype(np.float64)
            if approximate:
                phi[j, 0] = np.mean(np.log(count1 / emb_data1.shape[0]))
                phi[j, 1] = np.mean(np.log(count2 / emb_data2.shape[0]))
            else:
                phi[j, 0] = np.mean((count1 - 1) / (emb_data1.shape[0] - 1))
                phi[j, 1] = np.mean((count2 - 1) / (emb_data2.shape[0] - 1))
        return phi

    def sample_entropy(self, data, emb=2, metric='chebyshev'):
        """sample_entropy.
        https://en.wikipedia.org/wiki/Sample_entropy

        Args:
            data (np.ndarray): EEG data with the shape of (n_ch, n_points)

        Returns:
            np.ndarray: with the shape (n_ch, n_features per channel)
        """
        phi = self._app_samp_entropy_helper(data, emb=emb, metric=metric,
                                    approximate=False)
        if np.allclose(phi[:, 0], 0) or np.allclose(phi[:, 1], 0):
            raise ValueError('Sample Entropy is not defined.')
        else:
            return -np.log(np.divide(phi[:, 1], phi[:, 0]))

    def app_entropy(self, data, emb=2, metric='chebyshev'):
        """sample_entropy.
        https://en.wikipedia.org/wiki/Approximate_entropy

        Args:
            data (np.ndarray): EEG data with the shape of (n_ch, n_points)

        Returns:
            np.ndarray: with the shape (n_ch, n_features per channel)
        """
        phi = self._app_samp_entropy_helper(data, emb=emb, metric=metric,
                                    approximate=True)
        return np.subtract(phi[:, 0], phi[:, 1])

    def katz_fractal(self, data):
        """https://ieeexplore.ieee.org/document/8929940

        Args:
            data (np.ndarray): EEG data with the shape of (n_ch, n_points)

        Returns:
            np.ndarray: with the shape (n_ch, n_features per channel)
        """
        dists = np.abs(np.diff(data, axis=-1))
        ll = np.sum(dists, axis=-1)
        a = np.mean(dists, axis=-1)
        ln = np.log10(np.divide(ll, a))
        aux_d = data - data[:, 0, None]
        d = np.max(np.abs(aux_d[:, 1:]), axis=-1)
        katz = np.divide(ln, np.add(ln, np.log10(np.divide(d, ll))))
        return katz

    def _wavelet_coefs(self, data, wavelet_name='db4'):
        """Compute Discrete Wavelet Transform coefficients.

        Args:
            data (np.ndarray): EEG data with the shape of (n_ch, n_points)
            wavelet_name (str, optional):  Wavelet name (to be used with ``pywt.Wavelet``). The full list of
                                           Wavelet names are given by: ``[name for family in pywt.families() for
                                           name in pywt.wavelist(family)]``. Defaults to 'db4'.

        Returns:
            np.ndarray: Coefficients of a DWT (Discrete Wavelet Transform). ``coefs[0]`` is
                        the array of approximation coefficient and ``coefs[1:]`` is the list
                        of detail coefficients.
        """
        wavelet = pywt.Wavelet(wavelet_name)
        levdec = min(pywt.dwt_max_level(data.shape[-1], wavelet.dec_len), 6)
        coefs = pywt.wavedec(data, wavelet=wavelet, level=levdec)
        return coefs
        
    def wavelet_coef_energy(self, data, wavelet_name='db4'): #retyurns 6 datapoints, currently NOT to be used
        """Wavelet Coefficients energy

        Args:
            data (np.ndarray): EEG data with the shape of (n_ch, n_points)

        Returns:
            np.ndarray: with the shape (n_ch, n_features per channel)
        """
        n_channels = data.shape[0]
        coefs = self._wavelet_coefs(data, wavelet_name)
        levdec = len(coefs) - 1
        wavelet_energy = np.zeros((n_channels, levdec))
        for j in range(n_channels):
            for level in range(levdec):
                wavelet_energy[j, level] = np.sum(coefs[levdec - level][j, :] ** 2)
        return wavelet_energy

    def ptp_amp(self, data):
        """Wavelet Coefficients energy

        Args:
            data (np.ndarray): EEG data with the shape of (n_ch, n_points)

        Returns:
            np.ndarray: with the shape (n_ch, n_features per channel)
        """
        return np.ptp(data, axis=-1)

    def _slope_lstsq(self, x, y):
        """Slope of a 1D least-squares regression.
        Utility function which returns the slope of the linear regression
        between x and y.

        Args:
            x (np.ndarray): shape (n_times,)
            y (np.ndarray): shape (n_times,)

        Returns:
            float:
        """
        n_times = x.shape[0]
        sx2 = 0
        sx = 0
        sy = 0
        sxy = 0
        for j in range(n_times):
            sx2 += x[j] ** 2
            sx += x[j]
            sxy += x[j] * y[j]
            sy += y[j]
        den = n_times * sx2 - (sx ** 2)
        num = n_times * sxy - sx * sy
        return num / den

    def higuchi_fractal(self, data, kmax = 10):
        """higuchi_fracta.
        https://ieeexplore.ieee.org/document/8929940

        Args:
            data (np.ndarray): EEG data with the shape of (n_ch, n_points)

        Returns:
            np.ndarray: with the shape (n_ch, n_features per channel)
        """
        n_channels, n_times = data.shape
        higuchi = np.empty((n_channels,), dtype=data.dtype)
        for s in range(n_channels):
            kmax = np.int64(kmax)
            lk = np.empty((kmax,))
            x_reg = np.empty((kmax,))
            y_reg = np.empty((kmax,))
            for k in range(1, kmax + 1):
                lm = np.empty((k,))
                for m in range(k):
                    ll = 0
                    n_max = np.floor((n_times - m - 1) / k)
                    n_max = int(n_max)
                    for j in range(1, n_max):
                        ll += abs(data[s, m + j * k] - data[s, m + (j - 1) * k])
                    ll /= k
                    ll *= (n_times - 1) / (k * n_max)
                    lm[m] = ll
                # Mean of lm
                m_lm = 0
                for m in range(k):
                    m_lm += lm[m]
                m_lm /= k
                lk[k - 1] = m_lm
                x_reg[k - 1] = np.log(1. / k)
                y_reg[k - 1] = np.log(m_lm)
            higuchi[s] = self._slope_lstsq(x_reg, y_reg)
        return higuchi

    def zero_cross(self, data, threshold=np.finfo(np.float64).eps):
        """zero_cross

        Args:
            data (np.ndarray): EEG data with the shape of (n_ch, n_points)

        Returns:
            np.ndarray: with the shape (n_ch, n_features per channel)
        """
        _data = data.copy()
        # clip 'small' values to 0
        _data[np.abs(_data) < threshold] = 0
        sgn = np.sign(_data)
        # sgn may already contain 0 values (either 'true' zeros or clipped values)
        aux = np.diff((sgn == 0).astype(np.int64), axis=-1)
        count = np.sum(aux == 1, axis=-1) + (_data[:, 0] == 0)
        # zero between two consecutive time points (data[i] * data[i + 1] < 0)
        mask_implicit_zeros = sgn[:, 1:] * sgn[:, :-1] < 0
        count += np.sum(mask_implicit_zeros, axis=-1)
        return count

    # SPECTRAL BASED
    def psd(self, data, l_freq=4, h_freq=45, num_points=10):
        """Method to extract PSD of the signal as feature

        Args:
            data (_type_): _description_
            l_freq (int, optional): _description_. Defaults to 4.
            h_freq (int, optional): _description_. Defaults to 45.

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        delta = h_freq - l_freq
        raw_data = mne.io.RawArray(data, self.info, verbose=0, copy="both")
        # (#points-1) = #intervals,delta= h_freq-l_freq
        nfft = int(((num_points-1) * self.info["sfreq"]) / delta)
        small_delta = (self.info["sfreq"] / nfft)
        
        # step size between frequencies calculated by FFT
        # small_delta as close to delta/#intervals but not larger
        # o.w. we last point will be > h_freq so we would have (#points-1) points
        freqs_test = np.arange(nfft // 2 + 1) * small_delta
        estimate_n_points = ((freqs_test >= l_freq) & (freqs_test <= h_freq)).sum()

        ctr = 0
        while estimate_n_points != num_points:
            if ctr > 3:
                # means while loop is stuck and nfft+-1 doesn't produce correct number of points
                sampling_ratio = self.info["sfreq"]
                print(f"l_freq = {l_freq}")
                print(f"h_freq = {h_freq}")
                print(f"sfreq = {sampling_ratio}")
                print(f"nfft = {nfft}")
                print(f"small_delta = {small_delta}")
                print(f"freqs_test = {freqs_test}")
                print(f"estimate_n_points = {estimate_n_points}")
                raise ValueError(
                    "Calculating NFFT is stuck in a loop. Examine the parameters."
                )

            if estimate_n_points > num_points:
                nfft = nfft - 1
            else:
                nfft = nfft + 1
            ctr = ctr + 1
            freqs_test = np.arange(nfft // 2 + 1) * (self.info["sfreq"] / nfft)
            estimate_n_points = ((freqs_test >= l_freq) & (freqs_test <= h_freq)).sum()

        # calculating the actual psd
        psd, freqs = mne.time_frequency.psd_welch(
            raw_data,
            n_fft=nfft,
            n_overlap=0,
            n_per_seg=None,
            fmin=l_freq,
            fmax=h_freq,
            window="boxcar",
            verbose=False,  # talks too much
            picks="eeg",
        )

        if freqs.shape[0] != num_points:
            print("Yoink happened")
            print(freqs.shape)
            sampling_ratio = self.info["sfreq"]
            print(f"l_freq = {l_freq}")
            print(f"h_freq = {h_freq}")
            print(f"sfreq = {sampling_ratio}")
            print(f"nfft = {nfft}")
            print(f"freqs_test = {freqs_test}")
            print(f"estimate_n_points = {estimate_n_points}")
            raise ValueError("Something went wrong when calculating the psd.")

        return np.log(psd)

    def power_spectrum(self, sfreq, data, fmin=0., fmax=256., psd_method='welch',
                    welch_n_fft=256, welch_n_per_seg=None, welch_n_overlap=0,
                    verbose=False):
        _verbose = 40 * (1 - int(verbose))
        _fmin, _fmax = max(0, fmin), min(fmax, sfreq / 2)
        if psd_method == 'welch':
            _n_fft = min(data.shape[-1], welch_n_fft)
            return psd_array_welch(data, sfreq, fmin=_fmin, fmax=_fmax,
                                n_fft=_n_fft, verbose=_verbose,
                                n_per_seg=welch_n_per_seg,
                                n_overlap=welch_n_overlap)
        elif psd_method == 'multitaper':
            return psd_array_multitaper(data, sfreq, fmin=_fmin, fmax=_fmax,
                                        verbose=_verbose)
        elif psd_method == 'fft':
            n_times = data.shape[-1]
            m = np.mean(data, axis=-1)
            _data = data - m[..., None]
            spect = np.fft.rfft(_data, n_times)
            mag = np.abs(spect)
            freqs = np.fft.rfftfreq(n_times, 1. / sfreq)
            psd = np.power(mag, 2) / (n_times ** 2)
            psd *= 2.
            psd[..., 0] /= 2.
            if n_times % 2 == 0:
                psd[..., -1] /= 2.
            mask = np.logical_and(freqs >= _fmin, freqs <= _fmax)
            return psd[..., mask], freqs[mask]
        else:
            raise ValueError(f'The given method {str(psd_method)} is not implemented. Valid '
                            'methods for the computation of the PSD are: '
                            '`welch`, `fft` or `multitaper`.')

    def spectral_entropy(self, data, psd_method='welch'):
        """spectral_entropy

        Args:
            data (np.ndarray): EEG data with the shape of (n_ch, n_points)

        Returns:
            np.ndarray: with the shape (n_ch, n_features per channel)
        """
        power_sd, _ = self.power_spectrum(self.info["sfreq"], data, fmin = 4, fmax = 40, psd_method=psd_method)
        m = np.sum(power_sd, axis=-1)
        psd_norm = np.divide(power_sd[:, 1:], m[:, None])
        return -np.sum(np.multiply(psd_norm, np.log2(psd_norm)), axis=-1)

    def theta_band(self, data):
        """Low frequency theta band: 4-8 Hz.
        https://imotions.com/blog/neural-oscillations/

        Args:
            data (np.ndarray): EEG data with the shape of (n_ch, n_points)

        Returns:
            np.ndarray: with the shape (n_ch, n_features per channel)
        """
        band = self.bands["theta"]
        filt_data = mne.filter.filter_data(data, self.info["sfreq"],
                                           band[0], band[-1], verbose=False)

        return np.sum(filt_data ** 2, axis=-1)


    def alpha_band(self, data):
        """Full frequency alpha band: ~10 Hz.
        https://imotions.com/blog/neural-oscillations/

        Args:
            data (np.ndarray): EEG data with the shape of (n_ch, n_points)
        Returns:
            np.ndarray: with the shape (n_ch, n_features per channel)
        """
        band = self.bands["alpha"]
        filt_data = mne.filter.filter_data(data, self.info["sfreq"],
                                           band[0], band[-1], verbose=False)
    
        return np.sum(filt_data ** 2, axis=-1)
    
    def low_alpha_band(self, data):
        """Low frequency alpha band: 8-10 Hz.
        https://imotions.com/blog/neural-oscillations/

        Args:
            data (np.ndarray): EEG data with the shape of (n_ch, n_points)

        Returns:
            np.ndarray: with the shape (n_ch, n_features per channel)
        """
        band = self.bands["low_alpha"]
        filt_data = mne.filter.filter_data(data, self.info["sfreq"],
                                           band[0], band[-1], verbose=False)
    
        return np.sum(filt_data ** 2, axis=-1)
    
    def high_alpha_band(self, data):
        """High frequency alpha band: 10-12 Hz.
        https://imotions.com/blog/neural-oscillations/

        Args:
            data (np.ndarray): EEG data with the shape of (n_ch, n_points)

        Returns:
            np.ndarray: with the shape (n_ch, n_features per channel)
        """
        band = self.bands["high_alpha"]
        filt_data = mne.filter.filter_data(data, self.info["sfreq"],
                                           band[0], band[-1], verbose=False)
    
        return np.sum(filt_data ** 2, axis=-1)
    
    def beta_band(self, data):
        """Mid frequency beta band: ~12-30 Hz.
        https://imotions.com/blog/neural-oscillations/

        Args:
            data (np.ndarray): EEG data with the shape of (n_ch, n_points)

        Returns:
            np.ndarray: with the shape (n_ch, n_features per channel)
        """
        band = self.bands["beta"]
        filt_data = mne.filter.filter_data(data, self.info["sfreq"],
                                           band[0], band[-1], verbose=False)

        return np.sum(filt_data ** 2, axis=-1)


    def gamma_band(self, data):
        """high frequency theta band: >30 Hz.
        https://imotions.com/blog/neural-oscillations/

        Args:
            data (np.ndarray): EEG data with the shape of (n_ch, n_points)

        Returns:
            np.ndarray: with the shape (n_ch, n_features per channel)
        """
        band = self.bands["gamma"]
        filt_data = mne.filter.filter_data(data, self.info["sfreq"],
                                           band[0], band[-1], verbose=False)

        return np.sum(filt_data ** 2, axis=-1)

    def _tk_energy(self, data):
        """Teager-Kaiser Energy.
        Utility function for :func:`compute_taeger_kaiser_energy`.
        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times)
        Returns
        -------
        output : ndarray, shape (n_channels, n_times - 2)
        """
        n_channels, n_times = data.shape
        tke = np.empty((n_channels, n_times - 2), dtype=data.dtype)
        for j in range(n_channels):
            for i in range(1, n_times - 1):
                tke[j, i - 1] = data[j, i] ** 2 - data[j, i - 1] * data[j, i + 1]
        return tke

    def teager_kaiser_energy(self, data, wavelet_name='db4'):
        """teager_kaiser_energy
        https://www.sciencedirect.com/science/article/abs/pii/S1051200418300927

        Args:
            data (np.ndarray): EEG data with the shape of (n_ch, n_points)

        Returns:
            np.ndarray: with the shape (n_ch, n_features per channel)
        """
        n_channels = data.shape[0]
        coefs = self._wavelet_coefs(data, wavelet_name)
        levdec = len(coefs) - 1
        tke = np.empty((n_channels, levdec + 1, 2))
        for level in range(levdec + 1):
            tk_energy = self._tk_energy(coefs[level])
            tke[:, level, 0] = np.mean(tk_energy, axis=-1)
            tke[:, level, 1] = np.std(tk_energy, ddof=1, axis=-1)
        return tke.ravel()

    def hjorth_mobility_spect(self, data, normalize=False,
                                    psd_method='welch'):
        psd, freqs = self.power_spectrum(self.info["sfreq"], data, psd_method=psd_method)
        w_freqs = np.power(freqs, 2)
        mobility = np.sum(np.multiply(psd, w_freqs), axis=-1)
        if normalize:
            mobility = np.divide(mobility, np.sum(psd, axis=-1))
        return mobility

    def hjorth_complexity_spect(self, data, normalize=False, 
                                    psd_method='welch'):
        psd, freqs = self.power_spectrum(self.info["sfreq"], data, psd_method=psd_method)
        w_freqs = np.power(freqs, 4)
        complexity = np.sum(np.multiply(psd, w_freqs), axis=-1)
        if normalize:
            complexity = np.divide(complexity, np.sum(psd, axis=-1))
        return complexity

    # # BIVARIATE FEATRURES
    # def phase_lock_val(self, data):
    #     return mne_features.bivariate.compute_phase_lock_val(data)

    def first_difference(self, data):
        diff = np.diff(data, axis=1)
        diff = np.abs(diff)
        mean_diff = np.mean(diff, axis=1)
        return mean_diff

    def spect_bands(self, data):
        """Spectral bands from the data
        between 0 and the nyquist freq ( sfreq/2 )

        Args:
            data (np.ndarray):

        Returns:
            np.array: power at different bands
        """
        freq_bands = np.array(
            [  # [0, 4],  # delta ( we do not have)
                [4, 8],  # theta
                [8, 12],  # alpha
                [12, 32],  # beta
                [32, 45], # gamma
            ]
        )

        n_freq_bands = freq_bands.shape[0]
        band_energy = np.empty((data.shape[0], n_freq_bands))
        for j in range(n_freq_bands):
            filtered_data = \
                mne.filter.filter_data(data, sfreq=self.info["sfreq"], l_freq=freq_bands[j,0],
                                       h_freq=freq_bands[j,1], picks=None,
                                       fir_design='firwin')
            band_energy[:, j] = np.sum(filtered_data ** 2, axis=-1)
        return band_energy


    def a_b_ratio(self, data):
        """Alpha to beta ratio
        """
        alpha = self.alpha_band(data)
        beta = self.beta_band(data)

        return alpha/beta

    def b_g_ratio(self, data):
        """Beta to Gamma ratio
        """
        beta = self.beta_band(data)
        gamma = self.gamma_band(data)

        return beta/gamma

    def th_b_ratio(self, data):
        """Theta to Beta ratio
        """
        theta = self.theta_band(data)
        beta = self.beta_band(data)

        return theta/beta

    def brain_assymetry(self, data):
        """Calculates the band powers at different quesdrants

        Args:
            data (np.ndarray): _description_

        Returns:
            np.array: shape=(n_bands*n_quadrants)
        """
        # electrodes names and quadrant positions in the extended 10-20 scheme
        qadrants = np.array([['Fpz', 'Fp1',  # front_left
                              'AFz', 'AF3', 'AF7',
                              'Fz', 'F1', 'F3', 'F5', 'F7', 'F9',
                              'FCz', 'FC1', 'FC3', 'FC5', 'FC7', 'FC9',
                              'Cz', 'C1', 'C3', 'C5', 'T7', 'T9', 'A1'],

                             ['Fpz', 'Fp2',  # front_right
                              'AFz', 'AF4', 'AF8',
                              'Fz', 'F2', 'F4', 'F6', 'F8', 'F10',
                              'FCz', 'FC2', 'FC4', 'FC6', 'FC8', 'FC10',
                              'Cz', 'C2', 'C4', 'C6', 'T8', 'T10', 'A2'],

                             ['Cz', 'C1', 'C3', 'C5', 'T7', 'T9', 'A1',  # back_left
                              'CPz', 'CP1', 'CP3', 'CP5', 'TP7', 'TP9',
                              'Pz', 'P1', 'P3', 'P5', 'P7', 'P9',
                              'POz', 'PO3', 'PO7',
                              'Oz', 'O1'],

                             ['Cz', 'C2', 'C4', 'C6', 'T8', 'T10', 'A2',  # back_right
                              'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
                              'Pz', 'P2', 'P4', 'P6', 'P8', 'P10',
                              'POz', 'PO4', 'PO8',
                              'Oz', 'O2']])

        bands = np.array([
            #[0, 4],  # delta
            [4, 8],  # theta
            [8, 12],  # alpha
            [12, 32],  # beta
            [32, 45]])  # gamma

        raw = mne.io.RawArray(data, self.info, verbose=0, copy="both")
        bands_power = np.zeros((bands.shape[0], 4))

        for quadrant in range(qadrants.shape[0]):
            for i in range(bands_power.shape[0]):
                picks = np.intersect1d(qadrants[quadrant, :], self.info["ch_names"])
                raw_quadrant = raw.copy().pick_channels(picks)
                filtered_signal = raw_quadrant.filter(bands[i][0], bands[i][1])
                # filtered_signal.plot_psd()
                filtered_signal = filtered_signal["data"][0]
                bands_power[i, quadrant] = np.sum(filtered_signal) ** 2 / filtered_signal.shape[1]

        bands_power_flat = np.concatenate(bands_power).ravel()

        return bands_power_flat
