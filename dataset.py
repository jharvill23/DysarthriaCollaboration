from torch.utils import data
import joblib
from torch.nn.utils.rnn import pad_sequence
import torch
from utils import get_file_metadata_ctc_training, phone2class, class2phone, get_file_metadata
import yaml
from easydict import EasyDict as edict
import numpy as np
from augment_data import time_warp, freq_mask
import matplotlib.pyplot as plt

config = edict(yaml.load(open('config.yml'), Loader=yaml.SafeLoader))

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  # taken from https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
  def __init__(self, params):
        'Initialization'
        self.list_IDs = params['files']
        self.mode = params["mode"]
        # self.specaugment = params['specaugment']
        self.wordlist = params['metadata_help']['wordlist']
        self.dictionary = params['metadata_help']['dictionary']
        self.phones = params['metadata_help']['phones']
        self.p2c = phone2class(self.phones)
        self.c2p = class2phone(self.phones)

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Get the data item'
        file = self.list_IDs[index]
        if self.mode == 'train':
            metadata = get_file_metadata_ctc_training(file, wordlist=self.wordlist, dictionary=self.dictionary)
        elif self.mode == 'eval':
            metadata = get_file_metadata(file, wordlist=self.wordlist, dictionary=self.dictionary)
        """Load data"""
        phones_as_classes = [self.p2c[config.data.SOS_token]]
        [phones_as_classes.append(self.p2c[x]) for x in metadata['phones']]
        phones_as_classes.append(self.p2c[config.data.EOS_token])
        if self.mode == 'train':
            spectrogram = joblib.load(file)

            # if self.specaugment:
            #     try:
            #         time_warped = time_warp(spectrogram)
            #         # plt.subplot(211)
            #         # plt.imshow(features.T)
            #         # plt.subplot(212)
            #         # plt.imshow(time_warped.T)
            #         # plt.show()
            #         """Now we need to do frequency masking"""
            #         frequency_masked = freq_mask(time_warped)
            #         # plt.subplot(211)
            #         # plt.imshow(spectrogram.T)
            #         # plt.subplot(212)
            #         # plt.imshow(frequency_masked.T)
            #         # plt.show()
            #         spectrogram = frequency_masked
            #     except:
            #         """"""


            """Get input and target lengths"""
            input_length = spectrogram.shape[0]
            target_length = len(phones_as_classes)
            """Convert to tensors"""
            spectrogram = torch.from_numpy(spectrogram)
            phones_as_classes = torch.FloatTensor(phones_as_classes)
            input_length = torch.FloatTensor([input_length])
            target_length = torch.FloatTensor([target_length])

            return spectrogram, phones_as_classes, input_length, target_length, metadata
        elif self.mode == 'eval':
            """"""
            spectrogram = joblib.load(file)
            """Get input and target lengths"""
            input_length = spectrogram.shape[0]
            target_length = len(phones_as_classes)
            """Convert to tensors"""
            spectrogram = torch.from_numpy(spectrogram)
            phones_as_classes = torch.FloatTensor(phones_as_classes)
            input_length = torch.FloatTensor([input_length])
            target_length = torch.FloatTensor([target_length])

            return spectrogram, phones_as_classes, input_length, target_length, metadata

  def fix_tensor(self, x):
      x.requires_grad = True
      x = x.cuda()
      return x

  def collate(self, batch):
      spectrograms = [item[0] for item in batch]
      phones = [item[1] for item in batch]
      input_lengths = [item[2] for item in batch]
      target_lengths = [item[3] for item in batch]
      metadata = [item[4] for item in batch]
      """Extract dysarthric or normal from speaker"""
      # classes = np.identity(2)
      # speaker_type = [config.train.dys_class if 'C' not in x['speaker'] else config.train.normal_class for x in metadata]
      # speaker_type = np.asarray(speaker_type)
      # speaker_type = torch.from_numpy(speaker_type)
      # speaker_type = speaker_type.to(dtype=torch.long)
      # speaker_type = speaker_type.cuda()
      """"""
      spectrograms = pad_sequence(spectrograms, batch_first=True, padding_value=0)
      phones = pad_sequence(phones, batch_first=True, padding_value=self.p2c[config.data.EOS_token])
      input_lengths = torch.squeeze(torch.stack(input_lengths))
      target_lengths = torch.squeeze(torch.stack(target_lengths))
      spectrograms = self.fix_tensor(spectrograms)
      phones = self.fix_tensor(phones)
      # input_lengths = self.fix_tensor(input_lengths)
      # target_lengths = self.fix_tensor(target_lengths)

      return {"spectrograms": spectrograms, "phones": phones,
              "input_lengths": input_lengths, "target_lengths": target_lengths,
              "metadata": metadata}

  def collate_adv(self, batch):
      """Create dummy tensor of longest length to pad for adversary"""
      dummy_adv_phone_tensor = torch.from_numpy(np.zeros(shape=(15,)))  # 15 is max length of transcription sequence including SOS and EOS
      dummy_adv_phone_tensor = dummy_adv_phone_tensor.to(torch.float32)
      """Most everything else is the same"""
      spectrograms = [item[0] for item in batch]
      num_items = len(spectrograms)
      phones_ = [item[1] for item in batch]
      input_lengths = [item[2] for item in batch]
      target_lengths = [item[3] for item in batch]
      metadata = [item[4] for item in batch]
      """Extract dysarthric or normal from speaker"""
      # classes = np.identity(2)
      speaker_type = [config.train.dys_class if 'C' not in x['speaker'] else config.train.normal_class for x in metadata]
      speaker_type = np.asarray(speaker_type)
      speaker_type = torch.from_numpy(speaker_type)
      speaker_type = speaker_type.to(dtype=torch.long)
      speaker_type = speaker_type.cuda()
      """"""
      spectrograms = pad_sequence(spectrograms, batch_first=True, padding_value=0)
      phones = pad_sequence(phones_, batch_first=True, padding_value=self.p2c[config.data.EOS_token])
      adv_phones = phones_
      adv_phones.append(dummy_adv_phone_tensor)
      adv_phones = pad_sequence(adv_phones, batch_first=True, padding_value=self.p2c[config.data.EOS_token])
      #adv_phones = adv_phones[0:config.train.batch_size, :]
      adv_phones = adv_phones[0: num_items, :]
      # adv_phones_numpy = adv_phones.detach().cpu().numpy()  # for debugging purposes only
      input_lengths = torch.squeeze(torch.stack(input_lengths))
      target_lengths = torch.squeeze(torch.stack(target_lengths))
      spectrograms = self.fix_tensor(spectrograms)
      phones = self.fix_tensor(phones)
      adv_phones = self.fix_tensor(adv_phones)
      # input_lengths = self.fix_tensor(input_lengths)
      # target_lengths = self.fix_tensor(target_lengths)

      return {"spectrograms": spectrograms, "phones": phones,
              "input_lengths": input_lengths, "target_lengths": target_lengths,
              "metadata": metadata, "speaker_type": speaker_type,
              'adv_phones': adv_phones}

  def collate_eval(self, batch):
      spectrograms = [item[0] for item in batch]
      phones = [item[1] for item in batch]
      input_lengths = [item[2] for item in batch]
      target_lengths = [item[3] for item in batch]
      metadata = [item[4] for item in batch]
      spectrograms = pad_sequence(spectrograms, batch_first=True, padding_value=0)
      phones = pad_sequence(phones, batch_first=True, padding_value=self.p2c[config.data.EOS_token])
      input_lengths = torch.squeeze(torch.stack(input_lengths))
      target_lengths = torch.squeeze(torch.stack(target_lengths))
      spectrograms = self.fix_tensor(spectrograms)
      phones = self.fix_tensor(phones)

      return {"spectrograms": spectrograms, "phones": phones,
              "input_lengths": input_lengths, "target_lengths": target_lengths,
              "metadata": metadata}
