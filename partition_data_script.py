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

config = edict(yaml.load(open('config.yml'), Loader=yaml.SafeLoader))
partition = joblib.load('full_filenames_data_partition.pkl')
wordlist = joblib.load('wordlist.pkl')
dictionary = joblib.load('dict.pkl')

# unique_words = []
# with open('dict.txt') as f:
#     for l in f:
#         l = l.replace('\n', '')
#         word = l.split(' ')[0]
#         if word != '':
#             unique_words.append(word)
# unique_words = set(unique_words)
# stop = None

def get_conversion_pairs():
    pairs = []
    for source in config.data.conversion_source_speakers:
        for target in config.data.conversion_target_speakers:
            pairs.append({'source': source, 'target': target})
    return pairs

def get_types(files):
    types = []
    for file in files:
        metadata = utils.get_file_metadata(file, wordlist=wordlist, dictionary=dictionary)
        types.append(metadata['word'].upper())
    types = set(types)
    return types


"""For each global partition we want to collect the number of train and validation utts for seen data from dtw"""
dtw_dir = config.directories.dtw
pairs = get_conversion_pairs()
if not os.path.exists('partition_stats_seen_data.pkl'):
    partition_stats = {1: {}, 2: {}}
    for global_partition in [1, 2]:
        if global_partition == 1:
            root_dir = './exps/PARTITION_1_trial_2_attention_vc_model_training'
        elif global_partition == 2:
            root_dir = './exps/PARTITION_2_trial_1_attention_vc_model_training'
        partition = joblib.load('full_filenames_data_partition.pkl')
        partition = partition['partition_' + str(global_partition)]
        normal_files_in_seen_train_data = []
        normal_files_in_seen_val_data = []
        dys_files_in_seen_train_data = []
        dys_files_in_seen_val_data = []
        types_in_normal_seen_train_data = []
        types_in_normal_seen_val_data = []
        types_in_dys_seen_train_data = []
        types_in_dys_seen_val_data = []
        for pair in tqdm(pairs):
            dir_name = pair['source'] + '_to_' + pair['target']
            pair_partition = joblib.load(os.path.join(root_dir, dir_name, 'pair_partition.pkl'))
            for file in pair_partition['train']:
                """Do some stuff"""
                metadata = utils.get_file_metadata_vc(file, wordlist=wordlist, dictionary=dictionary)
                types_in_normal_seen_train_data.append(metadata['word'])
                types_in_dys_seen_train_data.append(metadata['word'])
                normal_files_in_seen_train_data.append(metadata['normal_filename'])
                dys_filename = metadata['normal_filename'].replace(metadata['normal_speaker'], metadata['dys_speaker'])
                dys_files_in_seen_train_data.append(dys_filename)
                stop = None

            for file in pair_partition['val']:
                """Do some stuff"""
                metadata = utils.get_file_metadata_vc(file, wordlist=wordlist, dictionary=dictionary)
                types_in_normal_seen_val_data.append(metadata['word'])
                types_in_dys_seen_val_data.append(metadata['word'])
                normal_files_in_seen_val_data.append(metadata['normal_filename'])
                dys_filename = metadata['normal_filename'].replace(metadata['normal_speaker'], metadata['dys_speaker'])
                dys_files_in_seen_val_data.append(dys_filename)
                stop = None


            # filepairs = collect_files(os.path.join(root_dir, dir_name))
            # for file in filepairs:
            #     filename = file.split('/')[-1]
            #     normal_filepath = 'features/' + filename.split('_')[0] + '_' + filename.split('_')[3] + '_' + filename.split('_')[4] + '_' + filename.split('_')[5]
            #     dys_filepath = 'features/' + filename.split('_')[2] + '_' + filename.split('_')[3] + '_' + filename.split('_')[4] + '_' + filename.split('_')[5]
            #     if normal_filepath in partition['seen']['filenames_only']['normal']['train'] or normal_filepath in partition['seen']['filenames_only']['normal']['val'] or normal_filepath in partition['seen']['filenames_only']['normal']['test']:
            #         normal_files_in_seen_data.append(normal_filepath)
            #     if dys_filepath in partition['seen']['filenames_only']['dys']['train'] or dys_filepath in partition['seen']['filenames_only']['dys']['val'] or dys_filepath in partition['seen']['filenames_only']['dys']['test']:
            #         dys_files_in_seen_data.append(dys_filepath)
        normal_files_in_seen_train_data = set(normal_files_in_seen_train_data)
        normal_files_in_seen_val_data = set(normal_files_in_seen_val_data)
        dys_files_in_seen_train_data = set(dys_files_in_seen_train_data)
        dys_files_in_seen_val_data = set(dys_files_in_seen_val_data)
        types_in_normal_seen_train_data = set(types_in_normal_seen_train_data)
        types_in_normal_seen_val_data = set(types_in_normal_seen_val_data)
        types_in_dys_seen_train_data = set(types_in_dys_seen_train_data)
        types_in_dys_seen_val_data = set(types_in_dys_seen_val_data)

        partition_stats[global_partition]['normal_train_files'] = normal_files_in_seen_train_data
        partition_stats[global_partition]['normal_val_files'] = normal_files_in_seen_val_data
        partition_stats[global_partition]['dys_train_files'] = dys_files_in_seen_train_data
        partition_stats[global_partition]['dys_val_files'] = dys_files_in_seen_val_data
        partition_stats[global_partition]['normal_train_types'] = types_in_normal_seen_train_data
        partition_stats[global_partition]['normal_val_types'] = types_in_normal_seen_val_data
        partition_stats[global_partition]['dys_train_types'] = types_in_dys_seen_train_data
        partition_stats[global_partition]['dys_val_types'] = types_in_dys_seen_val_data


        # normal_files_in_seen_data = set(normal_files_in_seen_data)
        # dys_files_in_seen_data = set(dys_files_in_seen_data)
        # partition_stats[global_partition]['normal_files_in_seen_data'] = normal_files_in_seen_data
        # partition_stats[global_partition]['dys_files_in_seen_data'] = dys_files_in_seen_data
    joblib.dump(partition_stats, 'partition_stats_seen_data.pkl')
else:
    """"""
    partition_stats = joblib.load('partition_stats_seen_data.pkl')
"""Now we need to find same stats for the unseen data"""
original_dict = joblib.load('dict.pkl')
stop = None
if not os.path.exists('unseen_stats.pkl'):
    partition_stats = {1: {}, 2: {}}
    for global_partition in [1, 2]:
        if global_partition == 1:
            root_dir = './exps/PARTITION_1_trial_2_attention_vc_model_training'
        elif global_partition == 2:
            root_dir = './exps/PARTITION_2_trial_1_attention_vc_model_training'
        partition = joblib.load('full_filenames_data_partition.pkl')
        partition = partition['partition_' + str(global_partition)]
        partition = partition['unseen']['filenames_only']
        normal_files_in_unseen_train_data = partition['normal']['train'] + partition['normal']['val']
        normal_files_in_unseen_val_data = partition['normal']['test']
        dys_files_in_unseen_train_data = partition['dys']['train']
        dys_files_in_unseen_val_data = partition['dys']['val']
        dys_files_in_unseen_test_data = partition['dys']['test']
        types_in_normal_unseen_train_data = get_types(normal_files_in_unseen_train_data)
        types_in_normal_unseen_val_data = get_types(normal_files_in_unseen_val_data)
        types_in_dys_unseen_train_data = get_types(dys_files_in_unseen_train_data)
        types_in_dys_unseen_val_data = get_types(dys_files_in_unseen_val_data)
        types_in_dys_unseen_test_data = get_types(dys_files_in_unseen_test_data)




        partition_stats[global_partition]['normal_train_files'] = normal_files_in_unseen_train_data
        partition_stats[global_partition]['normal_val_files'] = normal_files_in_unseen_val_data
        partition_stats[global_partition]['dys_train_files'] = dys_files_in_unseen_train_data
        partition_stats[global_partition]['dys_val_files'] = dys_files_in_unseen_val_data
        partition_stats[global_partition]['dys_test_files'] = dys_files_in_unseen_test_data
        partition_stats[global_partition]['normal_train_types'] = types_in_normal_unseen_train_data
        partition_stats[global_partition]['normal_val_types'] = types_in_normal_unseen_val_data
        partition_stats[global_partition]['dys_train_types'] = types_in_dys_unseen_train_data
        partition_stats[global_partition]['dys_val_types'] = types_in_dys_unseen_val_data
        partition_stats[global_partition]['dys_test_types'] = types_in_dys_unseen_test_data
        stop = None

original_dict = joblib.load('dict.pkl')
dict_words = set(key for key, value in original_dict.items())
all_normal_types_found = partition_stats[1]['normal_train_types'].union(partition_stats[2]['normal_train_types'])
all_dys_types_found = partition_stats[1]['dys_train_types'].union(partition_stats[2]['dys_train_types'])
difference_normal = dict_words.difference(all_normal_types_found)
difference_dys = dict_words.difference(all_dys_types_found)
stop = None
