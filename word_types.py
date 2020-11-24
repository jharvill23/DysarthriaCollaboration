import os
from tqdm import tqdm
import numpy as np
import joblib
import torch
import torch.nn as nn
import model
import yaml
from easydict import EasyDict as edict
import shutil
from preprocessing import collect_files
import utils
from dataset import Dataset
from torch.utils import data
from itertools import groupby
import json
from Levenshtein import distance as levenshtein_distance
import language_model
import multiprocessing
import concurrent.futures
import random
import scipy.stats as stats
import matplotlib.pyplot as plt

# spectrogram = joblib.load('features/M10_B1_UW3_M5.pkl')
# plt.imshow(spectrogram.T)
# plt.show()


config = edict(yaml.load(open('config.yml'), Loader=yaml.SafeLoader))
partition = joblib.load('full_filenames_data_partition.pkl')
wordlist = joblib.load('wordlist.pkl')
dictionary = joblib.load('dict.pkl')

# unique_items = []
# unique_keys = []
# keys = [key for key, item in wordlist.items()]
# key_value_matching_pairs = []
# for key in keys:
#     value = wordlist[key]
#     for inner_key, inner_value in wordlist.items():
#         if inner_key != key:
#             if inner_value == value:
#                 if key.split('_')[0] != inner_key.split('_')[0]:
#                     key_value_matching_pairs.append([key, inner_key, value])
# stop = None

"""This code is commented out just to save time for next part, you used this!"""
# unique_items = []
# unique_keys = {}
# for key, item in wordlist.items():
#     if 'UW' not in key:
#         key_part = key.split('_')[1]
#     else:
#         key_part = key
#     unique_items.append(item)
#     if key_part not in unique_keys:
#         unique_keys[key_part] = item
# unique_items = set(unique_items)
#
# item_plus_key = []
# for key, value in unique_keys.items():
#     item_plus_key.append([value.upper(), key])
# item_plus_key.sort()
#
# repeated_pairs = []
# word_to_key = {}
# for i in range(len(item_plus_key)):
#     word_to_key[item_plus_key[i][0]] = item_plus_key[i][1]
# for i in range(len(item_plus_key)-1):
#     if item_plus_key[i][0] == item_plus_key[i+1][0]:
#         repeated_pairs.append([item_plus_key[i], item_plus_key[i+1]])
#         print([item_plus_key[i], item_plus_key[i+1]])
#
# """Find number of uncommon words"""
# uncommon_words = []
# common_words = []
# for key, item in word_to_key.items():
#     if 'UW' in item:
#         uncommon_words.append(key)
#     else:
#         common_words.append(key)



all_files = collect_files(config.directories.features)
used_utterances = {}
used_utterances_list = []
used_speakers = config.data.conversion_source_speakers
used_speakers.extend(config.data.conversion_target_speakers)
for file in all_files:
    metadata = utils.get_file_metadata(file, wordlist=wordlist, dictionary=dictionary)
    if metadata['speaker'] in used_speakers:
        if metadata['speaker'] not in used_utterances:
            used_utterances[metadata['speaker']] = [file]
        else:
            used_utterances[metadata['speaker']].append(file)
        used_utterances_list.append(file)

for key, value in used_utterances.items():
    print(key + ': ' + str(len(value)))

common_words = []
uncommon_words = []
words = []
for file in all_files:
    file_type = file.split('/')[-1]
    word_type_indicator = file_type.split('_')[1] + '_' + file_type.split('_')[2]
    if 'UW' in word_type_indicator:
        uncommon_words.append(word_type_indicator)
    else:
        common_words.append(word_type_indicator)
    metadata = utils.get_file_metadata(file, wordlist=wordlist, dictionary=dictionary)
    words.append(metadata['word'].upper())
words = set(words)
uncommon_words = set(uncommon_words)
common_words = set(common_words)
stop = None