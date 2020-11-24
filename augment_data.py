import os
from tqdm import tqdm
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import joblib
import multiprocessing
import concurrent.futures
from dtwalign import dtw
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import scipy.interpolate as interp
from scipy.signal import butter, lfilter
from copy import deepcopy
import collections
from functools import partial
from preprocessing import collect_files
import yaml
from easydict import EasyDict as edict
import utils  # this creates a partition if one doesn't exist, but use existing one for consistency
import shutil
import train_attention_vc
from torch.utils import data
import train_dcgan_vc
from dcgan_dataset import Dataset, collate

attention_exp_path = './exps/PARTITION_2_trial_1_attention_vc_model_training'
dcgan_exp_path = './exps/PARTITION_2_trial_1_dcgan_vc_model_training'
attention_model_number = '150000'
dcgan_model_number = '10000'
# dcgan_exp_path = './exps/trial_4_dcgan_vc_one_model_fixed_iterations'
# dcgan_model_number = '10000'

config = edict(yaml.load(open('config.yml'), Loader=yaml.SafeLoader))
GLOBAL_PARTITION = 2

ATTENTION_AUGMENT = False  # Run these before SpecAugment and SMOTE
DCGAN_AUGMENT = False  # Run these before SpecAugment and SMOTE
SPEC_AUGMENT_ATTENTION = False
SPEC_AUGMENT_DCGAN = False
SMOTE = False
SEEN_AUGMENT = False
SPEC_AUGMENT_SEEN = False
UNSEEN_AUGMENT = False
SPEC_AUGMENT_UNSEEN = False
UNSEEN_NORMAL_AUGMENT = False
SPEC_AUGMENT_UNSEEN_NORMAL = False
ALL_DATA_SPECAUGMENT = True
ALL_DATA_AUGMENT_ATTENTION = True
ALL_DATA_AUGMENT_DCGAN = True
ALL_DATA_AUGMENT_FEATURES = False

WORDLIST = joblib.load('wordlist.pkl')
DICTIONARY = joblib.load('dict.pkl')
# SEEN_UNSEEN_WORDS = joblib.load('seen_unseen_words.pkl')

def multiprocess_augment_all_data(file):
    filename = (file['file']).split('/')[-1]
    target_dir = file['target_dir']
    features = joblib.load(file['file'])
    for i in range(config.spec_augment.all_data_num_augment_examples):
        try:
            """Now we need to do time warping, time and frequency masking"""
            time_warped = time_warp(features)
            # plt.subplot(211)
            # plt.imshow(features.T)
            # plt.subplot(212)
            # plt.imshow(time_warped.T)
            # plt.show()
            """Now we need to do frequency masking"""
            frequency_masked = freq_mask(time_warped)
            # plt.subplot(211)
            # plt.imshow(features.T)
            # plt.subplot(212)
            # plt.imshow(frequency_masked.T)
            # plt.show()
            dump_path = os.path.join(target_dir, filename[:-4] + '_specaugment_' + str(i) + '.pkl')
            joblib.dump(frequency_masked, dump_path)
        except:
            print(features.shape[0])
            dump_path = os.path.join(target_dir, filename[:-4] + '_specaugment_' + str(i) + '.pkl')
            joblib.dump(features, dump_path)
            stop = None

def multiprocess_augment_attention(file):
    filename = file.split('/')[-1]
    features = joblib.load(file)
    for i in range(config.spec_augment.num_augment_examples):
        """Now we need to do time warping, time and frequency masking"""
        time_warped = time_warp(features)
        # plt.subplot(211)
        # plt.imshow(features.T)
        # plt.subplot(212)
        # plt.imshow(time_warped.T)
        # plt.show()
        """Now we need to do frequency masking"""
        frequency_masked = freq_mask(time_warped)
        # plt.subplot(211)
        # plt.imshow(features.T)
        # plt.subplot(212)
        # plt.imshow(frequency_masked.T)
        # plt.show()
        dump_path = os.path.join(config.directories.spec_augmented_data,
                                 filename[:-4] + '_specaugment_' + str(i) + '.pkl')
        joblib.dump(frequency_masked, dump_path)

def multiprocess_augment_dcgan(file):
    filename = file.split('/')[-1]
    features = joblib.load(file)
    for i in range(config.spec_augment.num_augment_examples):
        """Now we need to do time warping, time and frequency masking"""
        time_warped = time_warp(features)
        # plt.subplot(211)
        # plt.imshow(features.T)
        # plt.subplot(212)
        # plt.imshow(time_warped.T)
        # plt.show()
        """Now we need to do frequency masking"""
        frequency_masked = freq_mask(time_warped)
        # plt.subplot(211)
        # plt.imshow(features.T)
        # plt.subplot(212)
        # plt.imshow(frequency_masked.T)
        # plt.show()
        dump_path = os.path.join(config.directories.spec_augmented_data_dcgan,
                                 filename[:-4] + '_specaugment_' + str(i) + '.pkl')
        joblib.dump(frequency_masked, dump_path)

def multiprocess_seen_augment(file):
    filename = file.split('/')[-1]
    features = joblib.load(file)
    try:
        for i in range(config.spec_augment.num_augment_examples_seen_data):
            """Now we need to do time warping, time and frequency masking"""
            time_warped = time_warp(features)
            # plt.subplot(211)
            # plt.imshow(features.T)
            # plt.subplot(212)
            # plt.imshow(time_warped.T)
            # plt.show()
            """Now we need to do frequency masking"""
            frequency_masked = freq_mask(time_warped)
            # plt.subplot(211)
            # plt.imshow(features.T)
            # plt.subplot(212)
            # plt.imshow(frequency_masked.T)
            # plt.show()
            dump_path = os.path.join(config.directories.spec_augmented_seen_data_dysarthric,
                                     filename[:-4] + '_specaugment_' + str(i) + '.pkl')
            joblib.dump(frequency_masked, dump_path)
    except:
        """"""

def multiprocess_unseen_augment(file):
    filename = file.split('/')[-1]
    features = joblib.load(file)
    try:
        for i in range(config.spec_augment.num_augment_examples_unseen_data):
            """Now we need to do time warping, time and frequency masking"""
            time_warped = time_warp(features)
            # plt.subplot(211)
            # plt.imshow(features.T)
            # plt.subplot(212)
            # plt.imshow(time_warped.T)
            # plt.show()
            """Now we need to do frequency masking"""
            frequency_masked = freq_mask(time_warped)
            # plt.subplot(211)
            # plt.imshow(features.T)
            # plt.subplot(212)
            # plt.imshow(frequency_masked.T)
            # plt.show()
            dump_path = os.path.join(config.directories.spec_augmented_unseen_data,
                                     filename[:-4] + '_specaugment_' + str(i) + '.pkl')
            joblib.dump(frequency_masked, dump_path)
    except:
        """"""

def multiprocess_unseen_normal_augment(file):
    filename = file.split('/')[-1]
    features = joblib.load(file)
    try:
        for i in range(config.spec_augment.num_augment_examples_unseen_normal_data):
            """Now we need to do time warping, time and frequency masking"""
            time_warped = time_warp(features)
            # plt.subplot(211)
            # plt.imshow(features.T)
            # plt.subplot(212)
            # plt.imshow(time_warped.T)
            # plt.show()
            """Now we need to do frequency masking"""
            frequency_masked = freq_mask(time_warped)
            # plt.subplot(211)
            # plt.imshow(features.T)
            # plt.subplot(212)
            # plt.imshow(frequency_masked.T)
            # plt.show()
            dump_path = os.path.join(config.directories.spec_augmented_unseen_normal_data,
                                     filename[:-4] + '_specaugment_' + str(i) + '.pkl')
            joblib.dump(frequency_masked, dump_path)
    except:
        """"""

def put_seen_features_in_new_dir():
    if not os.path.isdir(config.directories.seen_words_augmented_dysarthric_data):
        os.mkdir(config.directories.seen_words_augmented_dysarthric_data)
    speaker_pairs = [f.path for f in os.scandir(attention_exp_path) if f.is_dir()]
    for pair in speaker_pairs:
        pair_name = pair.split('/')[-1]
        normal_speaker = pair_name.split('_')[0]
        dys_speaker = pair_name.split('_')[2]
        model_path = os.path.join(pair, 'models', attention_model_number + '-G.ckpt')

        """First let's get all seen utterances from normal speaker"""
        potential_files = collect_files(os.path.join(config.directories.dtw, pair_name))
        correct_files = []
        for file in potential_files:
            metadata = utils.get_file_metadata_vc(file, wordlist=WORDLIST, dictionary=DICTIONARY)
            word = metadata['word'].upper()
            if word in SEEN_UNSEEN_WORDS['seen']:
                """We want to convert the original features (not dtw)"""
                filename = metadata['normal_filename']
                filename = filename.replace(normal_speaker, dys_speaker)
                original_path = os.path.join(config.directories.features, filename + '.pkl')
                if os.path.exists(original_path):
                    correct_files.append(original_path)

        print(model_path)
        for file in tqdm(correct_files):
            utterance_part = (file.split('/')[-1]).replace(pair_name, '')[:-4]
            modded_utterance_part = '_'.join(utterance_part.split('_')[1:])
            output = joblib.load(file)
            # src, tgt = solver.get_batch_transformer_augment(file)
            # solver.G = solver.G.eval()
            # converted_output = solver.G(src)
            # converted_output = np.squeeze(converted_output.detach().cpu().numpy())
            # plt.imshow(converted_output.T)
            # plt.show()
            """Filename convention for augmented data is DYS_utt_augment_pair"""
            """This way if we cut off everything after augment pair we get the original dys utterance"""
            """I think this will be easier for the eventual CTC training for recognition"""
            #utterance_name = dys_speaker + '_' + modded_utterance_part + '_augment_' + pair_name + '.pkl'
            utterance_name = file.split('/')[-1]
            dump_path = os.path.join(config.directories.seen_words_augmented_dysarthric_data, utterance_name)
            joblib.dump(output, dump_path)

def put_unseen_features_in_new_dir():
    if not os.path.isdir(config.directories.unseen_words_augmented_data):
        os.mkdir(config.directories.unseen_words_augmented_data)
    speaker_pairs = [f.path for f in os.scandir(attention_exp_path) if f.is_dir()]
    for pair in speaker_pairs:
        pair_name = pair.split('/')[-1]
        normal_speaker = pair_name.split('_')[0]
        dys_speaker = pair_name.split('_')[2]
        model_path = os.path.join(pair, 'models', attention_model_number + '-G.ckpt')

        """First let's get all seen utterances from normal speaker"""
        potential_files = collect_files(os.path.join(config.directories.dtw, pair_name))
        correct_files = []
        for file in potential_files:
            metadata = utils.get_file_metadata_vc(file, wordlist=WORDLIST, dictionary=DICTIONARY)
            word = metadata['word'].upper()
            if word in SEEN_UNSEEN_WORDS['unseen']:
                """We want the original features of the dysarthric speaker"""
                filename = metadata['normal_filename']
                filename = filename.replace(normal_speaker, dys_speaker)
                original_path = os.path.join(config.directories.features, filename + '.pkl')
                if os.path.exists(original_path):
                    correct_files.append(original_path)

        print(model_path)
        for file in tqdm(correct_files):
            utterance_part = (file.split('/')[-1]).replace(pair_name, '')[:-4]
            modded_utterance_part = '_'.join(utterance_part.split('_')[1:])
            output = joblib.load(file)
            # src, tgt = solver.get_batch_transformer_augment(file)
            # solver.G = solver.G.eval()
            # converted_output = solver.G(src)
            # converted_output = np.squeeze(converted_output.detach().cpu().numpy())
            # plt.imshow(converted_output.T)
            # plt.show()
            """Filename convention for augmented data is DYS_utt_augment_pair"""
            """This way if we cut off everything after augment pair we get the original dys utterance"""
            """I think this will be easier for the eventual CTC training for recognition"""
            #utterance_name = dys_speaker + '_' + modded_utterance_part + '_augment_' + pair_name + '.pkl'
            utterance_name = file.split('/')[-1]
            dump_path = os.path.join(config.directories.unseen_words_augmented_data, utterance_name)
            joblib.dump(output, dump_path)

def put_unseen_normal_features_in_new_dir():
    if not os.path.isdir(config.directories.unseen_normal_augmented_data):
        os.mkdir(config.directories.unseen_normal_augmented_data)
    speaker_pairs = [f.path for f in os.scandir(attention_exp_path) if f.is_dir()]
    for pair in speaker_pairs:
        pair_name = pair.split('/')[-1]
        normal_speaker = pair_name.split('_')[0]
        dys_speaker = pair_name.split('_')[2]
        model_path = os.path.join(pair, 'models', attention_model_number + '-G.ckpt')

        """First let's get all seen utterances from normal speaker"""
        potential_files = collect_files(os.path.join(config.directories.dtw, pair_name))
        correct_files = []
        for file in potential_files:
            metadata = utils.get_file_metadata_vc(file, wordlist=WORDLIST, dictionary=DICTIONARY)
            word = metadata['word'].upper()
            if word in SEEN_UNSEEN_WORDS['unseen']:
                """We want the original features of the normal speaker"""
                filename = metadata['normal_filename']
                # filename = filename.replace(normal_speaker, dys_speaker)
                original_path = os.path.join(config.directories.features, filename + '.pkl')
                if os.path.exists(original_path):
                    correct_files.append(original_path)

        print(model_path)
        for file in tqdm(correct_files):
            utterance_part = (file.split('/')[-1]).replace(pair_name, '')[:-4]
            modded_utterance_part = '_'.join(utterance_part.split('_')[1:])
            output = joblib.load(file)
            # src, tgt = solver.get_batch_transformer_augment(file)
            # solver.G = solver.G.eval()
            # converted_output = solver.G(src)
            # converted_output = np.squeeze(converted_output.detach().cpu().numpy())
            # plt.imshow(converted_output.T)
            # plt.show()
            """Filename convention for augmented data is DYS_utt_augment_pair"""
            """This way if we cut off everything after augment pair we get the original dys utterance"""
            """I think this will be easier for the eventual CTC training for recognition"""
            #utterance_name = dys_speaker + '_' + modded_utterance_part + '_augment_' + pair_name + '.pkl'
            utterance_name = file.split('/')[-1]
            dump_path = os.path.join(config.directories.unseen_normal_augmented_data, utterance_name)
            joblib.dump(output, dump_path)

def augment_attention():
    if GLOBAL_PARTITION == 1:
        if not os.path.isdir(config.directories.attention_augmented_data_1):
            os.mkdir(config.directories.attention_augmented_data_1)
    elif GLOBAL_PARTITION == 2:
        if not os.path.isdir(config.directories.attention_augmented_data_2):
            os.mkdir(config.directories.attention_augmented_data_2)

    speaker_pairs = [f.path for f in os.scandir(attention_exp_path) if f.is_dir()]
    partition = joblib.load('full_filenames_data_partition.pkl')
    partition = partition['partition_' + str(GLOBAL_PARTITION)]
    for pair in speaker_pairs:
        pair_name = pair.split('/')[-1]
        normal_speaker = pair_name.split('_')[0]
        dys_speaker = pair_name.split('_')[2]
        model_path = os.path.join(pair, 'models', attention_model_number + '-G.ckpt')
        """Load the model"""
        solver = train_attention_vc.Solver(pair_dir=pair, normal_speaker=normal_speaker, dys_speaker=dys_speaker)
        solver.restore_model(path=model_path)
        """Convert Normal unseen train and val utterances from normal speaker,
         then save into config.directories.attention_augmented_data"""
        """First let's get Normal unseen train, val and test utterances from normal speaker
        We will use train and val utterances for training, and test utterances for validation"""
        # potential_files = collect_files(os.path.join(config.directories.dtw, pair_name))
        correct_files_ = partition['unseen']['filenames_only']['normal']['train'].copy()
        correct_files_.extend(partition['unseen']['filenames_only']['normal']['val'].copy())
        correct_files_.extend(partition['unseen']['filenames_only']['normal']['test'].copy())
        correct_files = []
        for file in correct_files_:
            metadata = utils.get_file_metadata(file, wordlist=WORDLIST, dictionary=DICTIONARY)
            if metadata['speaker'] == normal_speaker:
                correct_files.append(file)
        # for file in potential_files:
        #     metadata = utils.get_file_metadata_vc(file, wordlist=WORDLIST, dictionary=DICTIONARY)
        #     word = metadata['word'].upper()
        #     if word in SEEN_UNSEEN_WORDS['unseen']:
        #         """We want to convert the original features (not dtw)"""
        #         original_path = os.path.join(config.directories.features, metadata['normal_filename'] + '.pkl')
        #         if os.path.exists(original_path):
        #             correct_files.append(original_path)

        print(model_path)
        for file in tqdm(correct_files):
            utterance_part = (file.split('/')[-1]).replace(pair_name, '')[:-4]
            modded_utterance_part = '_'.join(utterance_part.split('_')[1:])
            src, tgt = solver.get_batch_transformer_augment(file)
            solver.G = solver.G.eval()
            converted_output = solver.G(src)
            converted_output = np.squeeze(converted_output.detach().cpu().numpy())
            # plt.imshow(converted_output.T)
            # plt.show()
            """Filename convention for augmented data is DYS_utt_augment_pair"""
            """This way if we cut off everything after augment pair we get the original dys utterance"""
            """I think this will be easier for the eventual CTC training for recognition"""
            # utterance_name = dys_speaker + '_' + modded_utterance_part + '_augment_' + pair_name + '.pkl'
            utterance_name = utterance_part + '_augment_' + pair_name + '.pkl'
            if GLOBAL_PARTITION == 1:
                dump_path = os.path.join(config.directories.attention_augmented_data_1, utterance_name)
            elif GLOBAL_PARTITION == 2:
                dump_path = os.path.join(config.directories.attention_augmented_data_2, utterance_name)
            joblib.dump(converted_output, dump_path)

def augment_dcgan():
    if GLOBAL_PARTITION == 1:
        if not os.path.isdir(config.directories.dcgan_augmented_data_1):
            os.mkdir(config.directories.dcgan_augmented_data_1)
    elif GLOBAL_PARTITION == 2:
        if not os.path.isdir(config.directories.dcgan_augmented_data_2):
            os.mkdir(config.directories.dcgan_augmented_data_2)

    speaker_pairs = [f.path for f in os.scandir(dcgan_exp_path) if f.is_dir()]
    partition = joblib.load('full_filenames_data_partition.pkl')
    partition = partition['partition_' + str(GLOBAL_PARTITION)]
    for count, pair in enumerate(speaker_pairs):
        pair_name = pair.split('/')[-1]
        normal_speaker = pair_name.split('_')[0]
        dys_speaker = pair_name.split('_')[2]
        model_path = os.path.join(pair, 'models', dcgan_model_number + '-G.ckpt')
        """Load the model"""
        solver = train_dcgan_vc.Solver(pair_dir=pair, normal_speaker=normal_speaker, dys_speaker=dys_speaker)
        solver.restore_model(path=model_path)

        """Convert Normal unseen train and val utterances from normal speaker,
                 then save into config.directories.dcgan_augmented_data_1"""
        """First let's get Normal unseen train, val and test utterances from normal speaker
        We will use train and val utterances for training, and test utterances for validation"""
        # potential_files = collect_files(os.path.join(config.directories.dtw, pair_name))
        correct_files_ = partition['unseen']['filenames_only']['normal']['train'].copy()
        correct_files_.extend(partition['unseen']['filenames_only']['normal']['val'].copy())
        correct_files_.extend(partition['unseen']['filenames_only']['normal']['test'].copy())
        correct_files = []
        for file in correct_files_:
            metadata = utils.get_file_metadata(file, wordlist=WORDLIST, dictionary=DICTIONARY)
            if metadata['speaker'] == normal_speaker:
                correct_files.append(file)

        # potential_files = collect_files(os.path.join(config.directories.dtw, pair_name))
        # correct_files = []
        # for file in potential_files:
        #     metadata = utils.get_file_metadata_vc(file, wordlist=WORDLIST, dictionary=DICTIONARY)
        #     word = metadata['word'].upper()
        #     if word in SEEN_UNSEEN_WORDS['unseen']:
        #         """We want to convert the original features (not dtw)"""
        #         original_path = os.path.join(config.directories.features, metadata['normal_filename'] + '.pkl')
        #         if os.path.exists(original_path):
        #             correct_files.append(original_path)

        # """Run all training data for that pair, then save into config.directories.attention_augmented_data"""
        # train, val = solver.get_train_val(pair, normal_speaker, dys_speaker)
        # train_data = Dataset({'pairs': train, "mode": 'train'})
        # train_gen = data.DataLoader(train_data, batch_size=1, shuffle=True, collate_fn=collate, drop_last=True)
        print(model_path)
        print(count)

        for file in tqdm(correct_files):
            utterance_part = (file.split('/')[-1]).replace(pair_name, '')[:-4]
            modded_utterance_part = '_'.join(utterance_part.split('_')[1:])
            src = joblib.load(file)
            src = np.expand_dims(src, axis=0)  # adding batch dimension
            src = np.expand_dims(src, axis=0)  # adding dummy dimension for convolution, see dcgan_dataset collate
            src = torch.from_numpy(src)
            src = solver.to_gpu(src)
            # src, tgt = solver.get_batch_transformer_augment(file)
            solver.G = solver.G.eval()
            converted_output = solver.G(src)
            converted_output = np.squeeze(converted_output.detach().cpu().numpy())
            # plt.imshow(converted_output.T)
            # plt.show()
            """Filename convention for augmented data is DYS_utt_augment_pair"""
            """This way if we cut off everything after augment pair we get the original dys utterance"""
            """I think this will be easier for the eventual CTC training for recognition"""
            # utterance_name = dys_speaker + '_' + modded_utterance_part + '_DCGANaugment_' + pair_name + '.pkl'
            utterance_name = utterance_part + '_augment_' + pair_name + '.pkl'
            if GLOBAL_PARTITION == 1:
                dump_path = os.path.join(config.directories.dcgan_augmented_data_1, utterance_name)
            elif GLOBAL_PARTITION == 2:
                dump_path = os.path.join(config.directories.dcgan_augmented_data_2, utterance_name)
            joblib.dump(converted_output, dump_path)


        # for batch in tqdm(train_gen):
        #     normal = batch['normal']
        #     dys = batch['dys']
        #     file = batch['pair'][0]  # batch size 1 for inference
        #     utterance_part = (file.split('/')[-1]).replace(pair_name, '')[:-4]
        #     solver.G = solver.G.eval()
        #     converted_output = solver.G(normal)
        #     converted_output = np.squeeze(converted_output.detach().cpu().numpy())
        #     # plt.imshow(converted_output.T)
        #     # plt.show()
        #     """Filename convention for augmented data is DYS_utt_augment_pair"""
        #     """This way if we cut off everything after augment pair we get the original dys utterance"""
        #     """I think this will be easier for the eventual CTC training for recognition"""
        #     utterance_name = dys_speaker + utterance_part + '_DCGANaugment_' + pair_name + '.pkl'
        #     dump_path = os.path.join(config.directories.dcgan_augmented_data, utterance_name)
        #     joblib.dump(converted_output, dump_path)

def augment_spec_attention():
    """Note: We want to use the original spectrograms. But we use the filenames from attention so we have the same
    amount of augmented data for each approach."""
    if not os.path.isdir(config.directories.spec_augmented_data):
        os.mkdir(config.directories.spec_augmented_data)
    files_to_augment = collect_files(config.directories.attention_augmented_data)

    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for _ in tqdm(executor.map(multiprocess_augment_attention, files_to_augment)):
            """"""

    # for file in tqdm(files_to_augment):
    #     filename = file.split('/')[-1]
    #     features = joblib.load(file)
    #     for i in range(config.spec_augment.num_augment_examples):
    #         """Now we need to do time warping, time and frequency masking"""
    #         time_warped = time_warp(features)
    #         # plt.subplot(211)
    #         # plt.imshow(features.T)
    #         # plt.subplot(212)
    #         # plt.imshow(time_warped.T)
    #         # plt.show()
    #         """Now we need to do frequency masking"""
    #         frequency_masked = freq_mask(time_warped)
    #         # plt.subplot(211)
    #         # plt.imshow(features.T)
    #         # plt.subplot(212)
    #         # plt.imshow(frequency_masked.T)
    #         # plt.show()
    #         dump_path = os.path.join(config.directories.spec_augmented_data,
    #                                  filename[:-4] + '_specaugment_' + str(i) + '.pkl')
    #         joblib.dump(frequency_masked, dump_path)

def augment_all_data():
    feature_target_dir = config.directories.features_specaugmented
    feature_source_dir = config.directories.features
    if GLOBAL_PARTITION == 1:
        attention_target_dir = config.directories.attention_specaugmented_data_1
        dcgan_target_dir = config.directories.dcgan_specaugmented_data_1
        attention_source_dir = config.directories.attention_augmented_data_1
        dcgan_source_dir = config.directories.dcgan_augmented_data_1
    elif GLOBAL_PARTITION == 2:
        attention_target_dir = config.directories.attention_specaugmented_data_2
        dcgan_target_dir = config.directories.dcgan_specaugmented_data_2
        attention_source_dir = config.directories.attention_augmented_data_2
        dcgan_source_dir = config.directories.dcgan_augmented_data_2


    """Attention"""
    if ALL_DATA_AUGMENT_ATTENTION:
        if not os.path.isdir(attention_target_dir):
            os.mkdir(attention_target_dir)
        attention_files = collect_files(attention_source_dir)
        multiprocess_attention_files = []
        for file in attention_files:
            multiprocess_attention_files.append({'file': file, 'target_dir': attention_target_dir})
        # multiprocess_augment_all_data(multiprocess_attention_files[0])

        # """Debugging"""
        # for file in tqdm(multiprocess_attention_files):
        #     multiprocess_augment_all_data(file)

        with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            for _ in tqdm(executor.map(multiprocess_augment_all_data, multiprocess_attention_files)):
                """"""

    if ALL_DATA_AUGMENT_DCGAN:
        if not os.path.isdir(dcgan_target_dir):
            os.mkdir(dcgan_target_dir)
        """DCGAN"""
        dcgan_files = collect_files(dcgan_source_dir)
        multiprocess_dcgan_files = []
        for file in dcgan_files:
            multiprocess_dcgan_files.append({'file': file, 'target_dir': dcgan_target_dir})
        multiprocess_augment_all_data(multiprocess_dcgan_files[0])
        with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            for _ in tqdm(executor.map(multiprocess_augment_all_data, multiprocess_dcgan_files)):
                """"""

    if ALL_DATA_AUGMENT_FEATURES:
        if not os.path.isdir(feature_target_dir):
            os.mkdir(feature_target_dir)
        """Features"""
        feature_files = collect_files(feature_source_dir)
        multiprocess_feature_files = []
        for file in feature_files:
            multiprocess_feature_files.append({'file': file, 'target_dir': feature_target_dir})
        multiprocess_augment_all_data(multiprocess_feature_files[0])
        with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            for _ in tqdm(executor.map(multiprocess_augment_all_data, multiprocess_feature_files)):
                """"""
    stop = None


def augment_spec_dcgan():
    """Note: We want to use the original spectrograms. But we use the filenames from attention so we have the same
    amount of augmented data for each approach."""
    if not os.path.isdir(config.directories.spec_augmented_data_dcgan):
        os.mkdir(config.directories.spec_augmented_data_dcgan)
    files_to_augment = collect_files(config.directories.dcgan_augmented_data)

    # multiprocess_augment_dcgan(files_to_augment[0])

    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for _ in tqdm(executor.map(multiprocess_augment_dcgan, files_to_augment)):
            """"""

def augment_seen_data():
    """Note: We want to use the original spectrograms. But we use the filenames from attention so we have the same
        amount of augmented data for each approach."""
    if not os.path.isdir(config.directories.spec_augmented_seen_data_dysarthric):
        os.mkdir(config.directories.spec_augmented_seen_data_dysarthric)
    files_to_augment = collect_files(config.directories.seen_words_augmented_dysarthric_data)

    # multiprocess_seen_augment(files_to_augment[0])

    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for _ in tqdm(executor.map(multiprocess_seen_augment, files_to_augment)):
            """"""

def augment_unseen_data():
    """Note: We want to use the original spectrograms. But we use the filenames from attention so we have the same
        amount of augmented data for each approach."""
    if not os.path.isdir(config.directories.spec_augmented_unseen_data):
        os.mkdir(config.directories.spec_augmented_unseen_data)
    files_to_augment = collect_files(config.directories.unseen_words_augmented_data)

    # multiprocess_unseen_augment(files_to_augment[0])

    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for _ in tqdm(executor.map(multiprocess_unseen_augment, files_to_augment)):
            """"""

def augment_unseen_normal_data():
    """Note: We want to use the original spectrograms. But we use the filenames from attention so we have the same
        amount of augmented data for each approach."""
    if not os.path.isdir(config.directories.spec_augmented_unseen_normal_data):
        os.mkdir(config.directories.spec_augmented_unseen_normal_data)
    files_to_augment = collect_files(config.directories.unseen_normal_augmented_data)

    # multiprocess_unseen_normal_augment(files_to_augment[0])

    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for _ in tqdm(executor.map(multiprocess_unseen_normal_augment, files_to_augment)):
            """"""

def augment_smote():
    stop = None

def time_warp(spec):
    """Wanted to try torchaudio functions but had trouble with installation so just use your functions for now"""
    middle_index = round(spec.shape[0]/2)
    warp_start = round(np.random.uniform(low=config.spec_augment.W, high=spec.shape[0] - config.spec_augment.W))
    warp_distance = round(np.random.uniform(low=0, high=config.spec_augment.W))
    split = warp_start + warp_distance
    array1 = spec[0:middle_index]
    array2 = spec[middle_index:]
    array1_target = np.zeros(shape=(split, spec.shape[1]))
    array2_target = np.zeros(shape=(spec.shape[0]-split, spec.shape[1]))
    warped1 = stretch(target=array1_target, y=array1)
    warped2 = stretch(target=array2_target, y=array2)
    warped_spec = np.concatenate((warped1, warped2), axis=0)
    return warped_spec

def freq_mask(spec):
    f = np.random.randint(low=0, high=config.spec_augment.F)
    f_0 = np.random.randint(low=0, high=config.data.num_mels - f)
    spec[:, f_0:f_0 + f] = 0
    t = np.random.randint(low=0, high=config.spec_augment.T)
    t_0 = np.random.randint(low=0, high=spec.shape[0] - t)
    spec[t_0:t_0 + t, :] = 0
    return spec

def interpolate_resample(array, desired_length):
    # link: https://stackoverflow.com/questions/38064697/interpolating-a-numpy-array-to-fit-another-array
    arr1_interp = interp.interp1d(np.arange(array.shape[0]), array, kind='linear')
    new_array = arr1_interp(np.linspace(0, array.shape[0] - 1, desired_length))
    return new_array

def stretch(target, y):
    length = target.shape[0]
    stretch_factor = length/y.shape[0]
    y = y.T
    new_y = []
    for i in range(y.shape[0]):
        new_y.append(interpolate_resample(y[i], length))
    new_y = np.asarray(new_y)
    y = new_y.T
    return y

def main():
    if ATTENTION_AUGMENT:
        augment_attention()
    if DCGAN_AUGMENT:
        augment_dcgan()
    if SEEN_AUGMENT:
        put_seen_features_in_new_dir()
    if UNSEEN_AUGMENT:
        put_unseen_features_in_new_dir()
    if UNSEEN_NORMAL_AUGMENT:
        put_unseen_normal_features_in_new_dir()
    if SPEC_AUGMENT_UNSEEN_NORMAL:
        augment_unseen_normal_data()
    if SPEC_AUGMENT_SEEN:
        augment_seen_data()
    if SPEC_AUGMENT_UNSEEN:
        augment_unseen_data()
    if SPEC_AUGMENT_ATTENTION:
        augment_spec_attention()
    if SPEC_AUGMENT_DCGAN:
        augment_spec_dcgan()
    if SMOTE:
        augment_smote()
    if ALL_DATA_SPECAUGMENT:
        augment_all_data()



if __name__ == '__main__':
    main()