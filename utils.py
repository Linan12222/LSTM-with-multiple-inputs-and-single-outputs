import torch
import numpy as np
import random
import tqdm


def get_rolling_window_multistep(forecasting_length, interval_length, window_length, features, labels):
    output_features = np.zeros((1, features.shape[0], window_length))
    output_labels = np.zeros((1, 1, forecasting_length))
    if features.shape[1] != labels.shape[1]:
        assert 'cant process such data'
    else:
        output_features = np.zeros((1, features.shape[0], window_length))
        output_labels = np.zeros((1, 1, forecasting_length))
        for index in tqdm.tqdm(range(0, features.shape[1]-interval_length-window_length-forecasting_length+1), desc='data preparing'):
            output_features = np.concatenate((output_features, np.expand_dims(features[:, index:index+window_length], axis=0)))
            output_labels = np.concatenate((output_labels, np.expand_dims(labels[:, index+interval_length+window_length: index+interval_length+window_length+forecasting_length], axis=0)))
    output_features = output_features[1:, :, :]
    output_labels = output_labels[1:, :, :]
    return torch.from_numpy(output_features), torch.from_numpy(output_labels)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False



