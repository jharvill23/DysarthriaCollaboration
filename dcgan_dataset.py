from torch.utils import data
import joblib
import scipy.interpolate as interp
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
import torch

def collate(batch):
    normal = [item[0] for item in batch]
    dys = [item[1] for item in batch]
    normal = pad_sequence(normal, batch_first=True, padding_value=0)
    dys = pad_sequence(dys, batch_first=True, padding_value=0)
    normal.requires_grad = True
    dys.requires_grad = True
    normal = normal.cuda()
    dys = dys.cuda()
    """ Add the dummy dimension 1 for the convolutions """
    normal = torch.unsqueeze(normal, dim=1)
    dys = torch.unsqueeze(dys, dim=1)
    return {"normal": normal, "dys": dys}

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  # taken from https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
  def __init__(self, params):
        'Initialization'
        self.list_IDs = params['pairs']
        self.mode = params["mode"]

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        pair = self.list_IDs[index]
        """Load data"""
        if self.mode == 'train':
            datum = joblib.load(pair)
            normal_feats = datum['normal']
            dys_feats = datum['dys']
            normal_feats = torch.from_numpy(normal_feats)
            dys_feats = torch.from_numpy(dys_feats)
            return normal_feats, dys_feats
        elif self.mode == 'eval':
            feature_types = ['f0', 'mcep', 'bap']
            normal_feats = (joblib.load(pair['normal']))
            normal_feats['f0'] = np.expand_dims(normal_feats['f0'], axis=1)
            # normal_model_feats = normal_feats[self.feature_type]
            dys_feats = (joblib.load(pair['dys']))
            """Stretch all features to double length"""
            for type in feature_types:
                stretch_dummy = np.zeros(shape=(round(1.4 * normal_feats[type].shape[0]), normal_feats[type].shape[1]))
                normal_feats[type], _ = np.squeeze(stretch(normal_feats[type], stretch_dummy))
            # stretch_dummy =
            normal_model_mcep_feats = torch.from_numpy(normal_feats['mcep'])
            normal_model_bap_feats = torch.from_numpy(normal_feats['bap'])

            utt_ID = (pair['normal']).split('/')[-1][:-4]
            utt_ID = utt_ID.split('_')[1:]
            delimiter = '_'
            utt_ID = delimiter.join(utt_ID)
            return {'normal_feats': normal_feats, 'normal_model_mcep_feats': normal_model_mcep_feats,
                    'normal_model_bap_feats': normal_model_bap_feats, 'dys_feats': dys_feats,
                    'utt_ID': utt_ID}