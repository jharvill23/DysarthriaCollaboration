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

# area = stats.norm(0, 1).cdf(-2.041)
# P = area*2

partition = joblib.load('full_filenames_data_partition.pkl')

def gillick_test(R1, R2):
    """First turn Levenshtein distance into 1 if not zero to indicate misclassification
    Not really necessary but doing it to keep the code nice and understandable"""
    R1[np.nonzero(R1)] = 1
    R2[np.nonzero(R2)] = 1
    """Initialize n00, n01, n10, n11"""
    n00 = 0
    n01 = 0
    n10 = 0
    n11 = 0
    for r1, r2 in zip(R1, R2):
        """Zero is correct, one is incorrect"""
        if r1 == 0 and r2 == 0:
            n00 += 1
        elif r1 == 0 and r2 == 1:
            n01 += 1
        elif r1 == 1 and r2 == 0:
            n10 += 1
        elif r1 == 1 and r2 == 1:
            n11 += 1
    q_hat = n10/(n01 + n10)
    z = np.abs(q_hat - 0.5)/np.sqrt(1/(4*(n01 + n10)))
    p = 2 * stats.norm(0, 1).cdf(-z)
    return p

if not os.path.exists('stats_data.pkl'):

    data = {'attention': {'1': {}, '2': {}}, 'dcgan': {'1': {}, '2': {}},
            'oracle': {'1': {}, '2': {}}, 'lack': {'1': {}, '2': {}}, 'limited': {'1': {}, '2': {}}}

    """Global partition 1 final experiment directories"""
    data['attention']['1']['dir'] = './exps/PARTITION_1_trial_2_attention_vc_CTC_TRAINING'
    data['dcgan']['1']['dir'] = './exps/PARTITION_1_trial_2_dcgan_vc_CTC_TRAINING'
    data['oracle']['1']['dir'] = './exps/PARTITION_1_trial_2_Oracle_Baseline_CTC_TRAINING'
    data['lack']['1']['dir'] = './exps/PARTITION_1_trial_2_Lack_Baseline_CTC_TRAINING'
    data['limited']['1']['dir'] = './exps/PARTITION_1_trial_2_Limited_Baseline_CTC_TRAINING'

    """Global partition 2 final experiment directories"""
    data['attention']['2']['dir'] = './exps/PARTITION_2_trial_1_attention_vc_CTC_training'
    data['dcgan']['2']['dir'] = './exps/PARTITION_2_trial_1_dcgan_vc_CTC_training'
    data['oracle']['2']['dir'] = './exps/PARTITION_2_trial_1_Oracle_Baseline_CTC_training'
    data['lack']['2']['dir'] = './exps/PARTITION_2_trial_1_Lack_Baseline_CTC_training'
    data['limited']['2']['dir'] = './exps/PARTITION_2_trial_1_Limited_Baseline_CTC_training'

    phones = joblib.load('phones.pkl')
    p2c = utils.phone2class(phones)
    c2p = utils.class2phone(phones)

    no_lm_folder = 'predictions'
    lm_folder = 'language_model_predictions_test'
    for exp_type, exp_data in data.items():
        for partition_number, partition_data in exp_data.items():
            no_lm_dir = os.path.join(partition_data['dir'], no_lm_folder)
            lm_dir = os.path.join(partition_data['dir'], lm_folder)
            partition_data['no_lm_results'] = collect_files(no_lm_dir)
            partition_data['lm_results'] = collect_files(lm_dir)
            """Let's get the levenshtein distances"""
            partition_data['no_lm_scores'] = {}
            for file in partition_data['no_lm_results']:
                file_key = file.split('/')[-1]
                results = joblib.load(file)
                predicted_string = [p2c[x] for x in results['predicted_phones']]
                predicted_string = [chr(x) for x in predicted_string]
                predicted_string = ''.join(predicted_string)
                true_string = [p2c[x] for x in results['true_phones']]
                true_string = [chr(x) for x in true_string]
                true_string = ''.join(true_string)
                distance = levenshtein_distance(predicted_string, true_string)
                partition_data['no_lm_scores'][file_key] = distance

            partition_data['lm_scores'] = {}
            for file in partition_data['lm_results']:
                file_key = file.split('/')[-1]
                results = joblib.load(file)
                predicted_string = [p2c[x] for x in results['predicted_phones']]
                predicted_string = [chr(x) for x in predicted_string]
                predicted_string = ''.join(predicted_string)
                true_string = [p2c[x] for x in results['true_phones']]
                true_string = [chr(x) for x in true_string]
                true_string = ''.join(true_string)
                distance = levenshtein_distance(predicted_string, true_string)
                partition_data['lm_scores'][file_key] = distance

    joblib.dump(data, 'stats_data.pkl')

else:
    data = joblib.load('stats_data.pkl')
    if not os.path.exists('t_test_results.pkl'):
        """Make pairs for t-tests"""
        test_pairs = [['oracle', 'attention'], ['dcgan', 'attention'],
                      ['lack', 'attention'], ['limited', 'attention'], ['lack', 'dcgan']]

        p_values = {}
        gillick_p_values = {}
        for pair in test_pairs:
            against_method = pair[0]  # method we're comparing attention against
            attention_method = pair[1]
            pair_string = against_method + '_' + attention_method
            p_values[pair_string] = {}
            gillick_p_values[pair_string] = {}
            for score_type in ['no_lm_scores', 'lm_scores']:
                """Merge the dictionaries"""
                full_attention_dict = {**data[attention_method]['1'][score_type], **data[attention_method]['2'][score_type]}
                full_against_dict = {**data[against_method]['1'][score_type], **data[against_method]['2'][score_type]}
                score_attention_array = np.zeros(shape=(len(full_attention_dict),))
                score_against_array = np.zeros(shape=(len(full_against_dict),))
                i = 0
                for key, value in full_attention_dict.items():
                    score_attention_array[i] = value
                    score_against_array[i] = full_against_dict[key]
                    i += 1
                gillick_p_value = gillick_test(score_against_array, score_attention_array)
                t_test_results = stats.ttest_rel(score_against_array, score_attention_array)
                p_value = t_test_results.pvalue
                p_values[pair_string][score_type] = p_value
                gillick_p_values[pair_string][score_type] = gillick_p_value
        joblib.dump(p_values, 't_test_results.pkl')

    print(' & F05 & M14 & F04 & M05 & Mean \\\ \hline\hline')

    """Now we want to print WER table for latex"""
    for score_type in ['no_lm_scores', 'lm_scores']:
        for method in ['oracle', 'attention', 'dcgan', 'lack', 'limited']:
            speaker_data = {}
            full_method_dict = {**data[method]['1'][score_type], **data[method]['2'][score_type]}
            """Get mean for each speaker"""
            for filename, dist in full_method_dict.items():
                speaker = filename.split('_')[0]
                if speaker not in speaker_data:
                    speaker_data[speaker] = [dist]
                else:
                    speaker_data[speaker].append(dist)
            speaker_WERs = {}
            for speaker, distances in speaker_data.items():
                distances = np.asarray(distances)
                WER = 100*np.count_nonzero(distances)/distances.shape[0]
                speaker_WERs[speaker] = WER
            total_score = 0
            for key, value in speaker_WERs.items():
                total_score += value
            mean_WER = total_score/len(speaker_WERs)
            """Print the table for latex"""
            if score_type == 'no_lm_scores':
                if method == 'oracle' or method == 'limited':
                    print(method.title() + ' & ' + str(round(speaker_WERs['F05'], 2)) + ' & ' + str(
                        round(speaker_WERs['M14'], 2)) + ' & ' + str(round(speaker_WERs['F04'], 2)) + ' & ' + str(
                        round(speaker_WERs['M05'], 2)) + ' & ' + str(round(mean_WER, 2)) + '\\\ \hline\hline')
                elif method == 'attention':
                    print(method.title() + ' & ' + '{\\bf ' + str(
                        round(speaker_WERs['F05'], 2)) + ' } & ' + '{\\bf ' + str(
                        round(speaker_WERs['M14'], 2)) + ' } & ' + '{\\bf ' + str(
                        round(speaker_WERs['F04'], 2)) + ' } & ' + '{\\bf ' + str(
                        round(speaker_WERs['M05'], 2)) + '} ' + ' & {\\bf ' + str(round(mean_WER, 2)) + '} ' + '\\\ \hline')
                else:
                    print(method.title() + ' & ' + str(round(speaker_WERs['F05'], 2)) + ' & ' + str(
                        round(speaker_WERs['M14'], 2)) + ' & ' + str(round(speaker_WERs['F04'], 2)) + ' & ' + str(
                        round(speaker_WERs['M05'], 2)) + ' & ' + str(round(mean_WER, 2)) + '\\\ \hline')

            if score_type == 'lm_scores':
                if method == 'oracle':
                    print(method.title() + ' + LM' + ' & ' + str(round(speaker_WERs['F05'], 2)) + ' & ' + str(
                        round(speaker_WERs['M14'], 2)) + ' & ' + str(round(speaker_WERs['F04'], 2)) + ' & ' + str(
                        round(speaker_WERs['M05'], 2)) + ' & ' + str(round(mean_WER, 2)) + '\\\ \hline\hline')
                elif method == 'attention':
                    print(method.title() + ' + LM' + ' & ' + '{\\bf ' + str(round(speaker_WERs['F05'], 2)) + ' } & ' + '{\\bf ' + str(
                        round(speaker_WERs['M14'], 2)) + ' } & ' + '{\\bf ' + str(round(speaker_WERs['F04'], 2)) + ' } & ' + '{\\bf ' + str(
                        round(speaker_WERs['M05'], 2)) + '} ' + ' & {\\bf ' + str(round(mean_WER, 2)) + '} '+ '\\\ \hline')
                else:
                    print(method.title() + ' + LM' + ' & ' + str(round(speaker_WERs['F05'], 2)) + ' & ' + str(round(speaker_WERs['M14'], 2)) + ' & ' + str(round(speaker_WERs['F04'], 2)) + ' & ' + str(round(speaker_WERs['M05'], 2)) + ' & ' + str(round(mean_WER, 2)) + '\\\ \hline')

            stop = None




