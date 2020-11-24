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

config = edict(yaml.load(open('config.yml'), Loader=yaml.SafeLoader))

if not os.path.exists(config.directories.exps):
    os.mkdir(config.directories.exps)

trial = 'PARTITION_1_trial_2_Limited_Baseline_CTC_TRAINING'
exp_dir = os.path.join(config.directories.exps, trial)
if not os.path.isdir(exp_dir):
    os.mkdir(exp_dir)

GLOBAL_PARTITION = 1

TRAIN = False
EVAL = True
LOAD_MODEL = True
PERFORMANCE_EVAL = True
TUNE_LANGUAGE_MODEL = False
WORDSPLIT = False  # WORDSPLIT and UTTERANCESPLIT (one should be true and other false)
UTTERANCE_SPLIT = True
LANGUAGE_MODEL = True
BASELINE = True   # this is referred to as Limited Baseline in paper
ATTENTION = False
DCGAN = False
ORIGINAL_UNSEEN_BASELINE = False  # this is referred to as Oracle Baseline in paper
UNSEEN_NORMAL_BASELINE = False  # this is referred to as Lack Baseline in paper

RESUME_TRAINING = False
if RESUME_TRAINING:
    LOAD_MODEL = True

WORDLIST = joblib.load('wordlist.pkl')
DICTIONARY = joblib.load('dict.pkl')

def multiprocessing_lm_eval(data):
    file = data['file']
    language_models = data['lm']
    c2p = data['c2p']
    p2c = data['p2c']

    filename = file.split('/')[-1]
    data = joblib.load(file)
    outputs = data['ctc_outputs']

    """You need to add language models up to the longest sequence length dude"""
    best_sequence = language_model.get_best_sequence(ctc_output=outputs,
                                                     language_models=language_models,
                                                     c2p=c2p, p2c=p2c, true_transcription='')

    predicted_phones_ = best_sequence
    """remove SOS and EOS"""
    predicted_phones = []
    for x in predicted_phones_:
        if x != 'SOS' and x != 'EOS':
            predicted_phones.append(x)

    data_to_save = {'speaker': data['speaker'],
                    'word': data['word'],
                    'true_phones': data['true_phones'],
                    'predicted_phones': predicted_phones}
    dump_path = os.path.join(exp_dir, 'language_model_predictions', filename)
    joblib.dump(data_to_save, dump_path)

class Solver(object):
    """Solver"""

    def __init__(self):
        """Initialize configurations."""

        # Training configurations.
        self.g_lr = config.model.lr
        self.torch_type = torch.float32

        # Miscellaneous.
        self.use_tensorboard = True
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:{}'.format(0) if self.use_cuda else 'cpu')

        # Directories.
        self.log_dir = os.path.join(exp_dir, 'logs')
        self.model_save_dir = os.path.join(exp_dir, 'models')
        self.train_data_dir = config.directories.features
        self.predict_dir = os.path.join(exp_dir, 'predictions')

        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)
        if not os.path.isdir(self.model_save_dir):
            os.mkdir(self.model_save_dir)
        if not os.path.isdir(self.predict_dir):
            os.mkdir(self.predict_dir)

        self.partition = 'full_filenames_data_partition.pkl'
        """Partition file"""
        if TRAIN:  # only copy these when running a training session, not eval session
            # if not os.path.exists('ctc_partition.pkl'):
            #     utils.get_ctc_partition_offline_augmentation()
            # if not os.path.exists('baseline_ctc_partition.pkl'):
            #     utils.get_baseline_ctc_partition_offline_augmentation()
            # if not os.path.exists('dcgan_ctc_partition.pkl'):
            #     utils.get_dcgan_ctc_partition_offline_augmentation()
            # if not os.path.exists('original_unseen_ctc_partition.pkl'):
            #     utils.get_original_unseen_ctc_partition_offline_augmentation()
            # if not os.path.exists('unseen_normal_ctc_partition.pkl'):
            #     utils.get_unseen_normal_ctc_partition_offline_augmentation()
            # copy partition to exp_dir then use that for trial (just in case you change partition for other trials)
            # if BASELINE:
            #     shutil.copy(src='baseline_ctc_partition.pkl', dst=os.path.join(exp_dir, 'baseline_ctc_partition.pkl'))
            #     self.partition = os.path.join(exp_dir, 'baseline_ctc_partition.pkl')
            # elif ATTENTION:
            #     shutil.copy(src='ctc_partition.pkl', dst=os.path.join(exp_dir, 'ctc_partition.pkl'))
            #     self.partition = os.path.join(exp_dir, 'ctc_partition.pkl')
            # elif DCGAN:
            #     shutil.copy(src='dcgan_ctc_partition.pkl', dst=os.path.join(exp_dir, 'dcgan_ctc_partition.pkl'))
            #     self.partition = os.path.join(exp_dir, 'dcgan_ctc_partition.pkl')
            # elif ORIGINAL_UNSEEN_BASELINE:
            #     shutil.copy(src='original_unseen_ctc_partition.pkl', dst=os.path.join(exp_dir, 'original_unseen_ctc_partition.pkl'))
            #     self.partition = os.path.join(exp_dir, 'original_unseen_ctc_partition.pkl')
            # elif UNSEEN_NORMAL_BASELINE:
            #     shutil.copy(src='unseen_normal_ctc_partition.pkl',
            #                 dst=os.path.join(exp_dir, 'unseen_normal_ctc_partition.pkl'))
            #     self.partition = os.path.join(exp_dir, 'unseen_normal_ctc_partition.pkl')
            shutil.copy(src='full_filenames_data_partition.pkl',
                        dst=os.path.join(exp_dir, 'full_filenames_data_partition.pkl'))
            # copy config as well
            shutil.copy(src='config.yml', dst=os.path.join(exp_dir, 'config.yml'))
            # copy dict
            shutil.copy(src='dict.pkl', dst=os.path.join(exp_dir, 'dict.pkl'))
            # copy phones
            shutil.copy(src='phones.pkl', dst=os.path.join(exp_dir, 'phones.pkl'))
            # copy wordlist
            shutil.copy(src='wordlist.pkl', dst=os.path.join(exp_dir, 'wordlist.pkl'))

        # if TUNE_LANGUAGE_MODEL:
        #     self.partition = os.path.join(exp_dir, 'baseline_ctc_partition.pkl')

        if LANGUAGE_MODEL:
            print('Loading language models...')
            self.language_models = language_model.get_language_model()
            # self.model_1gram = language_models['model_1gram']
            # self.model_2gram = language_models['model_2gram']
            # self.model_3gram = language_models['model_3gram']
            # self.model_4gram = language_models['model_4gram']
            # self.model_5gram = language_models['model_5gram']
            # self.model_6gram = language_models['model_6gram']
            print('Language models loaded.')
            self.predict_dir = os.path.join(exp_dir, 'language_model_predictions_test')

        # Step size.
        self.log_step = config.train.log_step
        self.model_save_step = config.train.model_save_step

        # Build the model
        self.build_model()
        if EVAL or LOAD_MODEL:
            self.restore_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Build the model"""
        self.G = model.CTCmodel(config)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr)
        self.print_network(self.G, 'G')
        self.G.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def print_optimizer(self, opt, name):
        print(opt)
        print(name)

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def _load(self, checkpoint_path):
        if self.use_cuda:
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path,
                                    map_location=lambda storage, loc: storage)
        return checkpoint

    def restore_model(self):
        """Restore the model"""
        print('Loading the trained models... ')
        # G_path = './exps/PARTITION_1_trial_1_attention_vc_CTC_TRAINING/models/310000-G.ckpt'
        if GLOBAL_PARTITION == 1:
            if ATTENTION:
                G_path = './exps/PARTITION_1_trial_2_attention_vc_CTC_TRAINING/models/460000-G.ckpt'
            if DCGAN:
                G_path = './exps/PARTITION_1_trial_2_dcgan_vc_CTC_TRAINING/models/585000-G.ckpt'
            if BASELINE:
                G_path = './exps/PARTITION_1_trial_2_Limited_Baseline_CTC_TRAINING/models/260000-G.ckpt'
            if ORIGINAL_UNSEEN_BASELINE:
                G_path = './exps/PARTITION_1_trial_2_Oracle_Baseline_CTC_TRAINING/models/305000-G.ckpt'
                # G_path = './exps/PARTITION_1_trial_3_Oracle_Baseline_CTC_TRAINING/models/230000-G.ckpt'
            if UNSEEN_NORMAL_BASELINE:
                G_path = './exps/PARTITION_1_trial_2_Lack_Baseline_CTC_TRAINING/models/420000-G.ckpt'
        elif GLOBAL_PARTITION == 2:
            if ATTENTION:
                G_path = './exps/PARTITION_2_trial_1_attention_vc_CTC_training/models/370000-G.ckpt'
            if DCGAN:
                G_path = './exps/PARTITION_2_trial_1_dcgan_vc_CTC_training/models/470000-G.ckpt'
            if ORIGINAL_UNSEEN_BASELINE:
                G_path = './exps/PARTITION_2_trial_1_Oracle_Baseline_CTC_training/models/240000-G.ckpt'
            if BASELINE:
                G_path = './exps/PARTITION_2_trial_1_Limited_Baseline_CTC_training/models/260000-G.ckpt'
            if UNSEEN_NORMAL_BASELINE:
                G_path = './exps/PARTITION_2_trial_1_Lack_Baseline_CTC_training/models/415000-G.ckpt'
            if RESUME_TRAINING:
                if UNSEEN_NORMAL_BASELINE:
                    G_path = './exps/PARTITION_2_trial_1_Lack_Baseline_CTC_training/models/170000-G.ckpt'
        g_checkpoint = self._load(G_path)
        self.G.load_state_dict(g_checkpoint['model'])
        self.g_optimizer.load_state_dict(g_checkpoint['optimizer'])
        self.g_lr = self.g_optimizer.param_groups[0]['lr']

    def update_lr(self, g_lr):
        """Decay learning rates of g"""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()

    def wordlist_to_dict(self, words):
        train = {}
        test = {}
        for value in words['train']:
            train[value] = ''
        for value in words['test']:
            test[value] = ''
        return {'train': train, 'test': test}

    def filter_speakers(self, files):
        """"""
        new_files = []
        speakers = []
        for file in tqdm(files):
            speaker = (file.split('/')[-1]).split('_')[0]
            if speaker not in config.data.ignore_speakers:
                new_files.append(file)
                speakers.append(speaker)

        # speakers = set(speakers)  # check that it worked properly
        return new_files

    def get_train_test_wordsplit(self):
        wordlist = joblib.load('wordlist.pkl')
        dictionary = joblib.load('dict.pkl')
        """Get the triain and test files"""
        if not os.path.exists(os.path.join(exp_dir, 'train_test_files.pkl')):
            words = joblib.load(self.partition)
            """Turn train and test into dicts for fast check (hashable)"""
            words = self.wordlist_to_dict(words)
            train_files = []
            test_files = []
            for file in tqdm(collect_files(self.train_data_dir)):
                metadata = utils.get_file_metadata(file, wordlist=wordlist, dictionary=dictionary)
                try:
                    dummy = words['train'][metadata['word']]
                    train_files.append(file)
                except:
                    try:
                        dummy = words['test'][metadata['word']]
                        test_files.append(file)
                    except:
                        print("File in neither train nor test set...")
            joblib.dump({'train': train_files, 'test': test_files}, os.path.join(exp_dir, 'train_test_files.pkl'))
        else:
            files = joblib.load(os.path.join(exp_dir, 'train_test_files.pkl'))
            train_files = files['train']
            test_files = files['test']
        return self.filter_speakers(train_files), self.filter_speakers(test_files)

    def get_augmented_names_vc_data(self, filelist, augment_dir):
        new_list = []
        for file in filelist:
            metadata = utils.get_file_metadata(file, wordlist=WORDLIST, dictionary=DICTIONARY)
            normal_speaker = metadata['speaker']
            for speaker in config.data.conversion_target_speakers:
                for i in range(config.spec_augment.all_data_num_augment_examples):
                    new_filename = file.split('/')[-1][:-4]
                    new_filename = new_filename + '_augment_' + normal_speaker + '_to_' + speaker + '_specaugment_' + str(i) + '.pkl'
                    new_path = os.path.join(augment_dir, new_filename)
                    if os.path.exists(new_path):
                        new_list.append(new_path)
        return new_list

    def get_augmented_names_non_vc_data(self, filelist, augment_dir):
        # augmented_files = collect_files(config.directories.features_specaugmented)
        new_list = []
        for file in filelist:
            for i in range(config.spec_augment.all_data_num_augment_examples):
                new_filename = file.split('/')[-1][:-4]
                new_filename = new_filename + '_specaugment_' + str(i) + '.pkl'
                new_path = os.path.join(augment_dir, new_filename)
                if os.path.exists(new_path):
                    new_list.append(new_path)
        return new_list

    def get_true_test_utterances(self):
        partition = joblib.load(self.partition)
        partition = partition['partition_' + str(GLOBAL_PARTITION)]

        # """Debugging"""
        test_files = partition['unseen']['filenames_only']['dys']['test']
        # test_files.extend(partition['unseen']['filenames_only']['dys']['val'])
        # test_files.extend(partition['unseen']['filenames_only']['dys']['train'])

        return test_files

    def get_train_test_utterance_split(self):
        partition = joblib.load(self.partition)
        partition = partition['partition_' + str(GLOBAL_PARTITION)]
        if ATTENTION:
            partition = partition['unseen']['filenames_only']['normal']
            normal_train_files = partition['train'].copy()
            normal_train_files.extend(partition['val'].copy())
            normal_val_files = partition['test']
            if GLOBAL_PARTITION == 1:
                augment_dir = config.directories.attention_specaugmented_data_1
            elif GLOBAL_PARTITION == 2:
                augment_dir = config.directories.attention_specaugmented_data_2

        if DCGAN:
            partition = partition['unseen']['filenames_only']['normal']
            normal_train_files = partition['train'].copy()
            normal_train_files.extend(partition['val'].copy())
            normal_val_files = partition['test']
            if GLOBAL_PARTITION == 1:
                augment_dir = config.directories.dcgan_specaugmented_data_1
            elif GLOBAL_PARTITION == 2:
                augment_dir = config.directories.dcgan_specaugmented_data_2

        if ORIGINAL_UNSEEN_BASELINE:
            """Use the unseen train partition for training, unseen val partition for validation
            Remember model is tested on unseen test partition"""
            partition = partition['unseen']['filenames_only']['dys']
            pre_specaugment_train_files = partition['train'].copy()
            pre_specaugment_val_files = partition['val']
            augment_dir = config.directories.features_specaugmented

        if UNSEEN_NORMAL_BASELINE:
            partition = partition['unseen']['filenames_only']['normal']
            pre_specaugment_train_files = partition['train'].copy()
            pre_specaugment_val_files = partition['val'].copy()
            augment_dir = config.directories.features_specaugmented

        if BASELINE:
            """Use the seen train and val partitions for training, seen test partition for validation
                        Remember model is tested on unseen data"""
            partition = partition['seen']['filenames_only']['dys']
            pre_specaugment_train_files = partition['train'].copy()
            pre_specaugment_train_files.extend(partition['val'].copy())
            pre_specaugment_val_files = partition['test']
            augment_dir = config.directories.features_specaugmented


        if ATTENTION or DCGAN:
            """Now you need to convert normal filenames into the augmented names"""
            train_files = self.get_augmented_names_vc_data(normal_train_files, augment_dir)
            val_files = self.get_augmented_names_vc_data(normal_val_files, augment_dir)
        if ORIGINAL_UNSEEN_BASELINE or BASELINE or UNSEEN_NORMAL_BASELINE:
            train_files = self.get_augmented_names_non_vc_data(pre_specaugment_train_files, augment_dir)
            val_files = self.get_augmented_names_non_vc_data(pre_specaugment_val_files, augment_dir)

        # train_files = self.filter_speakers(partition["train"])
        # val_files = self.filter_speakers(partition["val"])
        return train_files, val_files

    def val_loss(self, val, iterations):
        """Time to write this function"""
        self.val_history = {}
        val_loss_value = 0
        for batch_number, features in tqdm(enumerate(val)):
            spectrograms = features['spectrograms']
            phones = features['phones']
            input_lengths = features['input_lengths']
            target_lengths = features['target_lengths']
            metadata = features["metadata"]
            # batch_speakers = [x['speaker'] for x in metadata]
            self.G = self.G.eval()

            """Make input_lengths and target_lengths torch ints"""
            input_lengths = input_lengths.to(torch.int32)
            target_lengths = target_lengths.to(torch.int32)
            phones = phones.to(torch.int32)

            spectrograms = spectrograms.to(self.torch_type)

            outputs = self.G(spectrograms)

            outputs = outputs.permute(1, 0, 2)  # swap batch and sequence length dimension for CTC loss

            loss = self.ctc_loss(log_probs=outputs, targets=phones,
                                 input_lengths=input_lengths, target_lengths=target_lengths)

            val_loss_value += loss.item()
            """Update the loss history MUST BE SEPARATE FROM TRAINING"""
            # self.update_history_val(loss, batch_speakers)
        """We have the history, now do something with it"""
        # val_loss_means = {}
        # for key, value in self.val_history.items():
        #     val_loss_means[key] = np.mean(np.asarray(value))
        # val_loss_means_sorted = {k: v for k, v in sorted(val_loss_means.items(), key=lambda item: item[1])}
        # weights = {}
        # counter = 1
        # val_loss_value = 0
        # for key, value in val_loss_means_sorted.items():
        #     val_loss_value += (config.train.fairness_lambda * counter + (1-config.train.fairness_lambda) * 1) * value
        #     counter += 1

        return val_loss_value

    def update_history(self, loss, speakers):
        """Update the history with the new loss values"""
        loss_copy = loss.detach().cpu().numpy()
        for loss_value, speaker in zip(loss_copy, speakers):
            speaker_index = self.s2i[speaker]
            """Extract row corresponding to speaker"""
            history_row = self.history[speaker_index]
            """Shift all elements by 1 to the right"""
            history_row = np.roll(history_row, shift=1)
            """Overwrite the first value (the last value in the array rolled to the front and is overwritten"""
            history_row[0] = loss_value
            """Set the history row equal to the modified row"""
            self.history[speaker_index] = history_row

    def update_history_val(self, loss, speakers):
        """Update the val_history with the new loss values"""
        loss_copy = loss.detach().cpu().numpy()
        for loss_value, speaker in zip(loss_copy, speakers):
            speaker_index = self.s2i[speaker]
            if speaker_index not in self.val_history:
                self.val_history[speaker_index] = []
            self.val_history[speaker_index].append(loss_value)

    def get_loss_weights(self, speakers, type='fair'):
        """Use self.history to determine the ranking of which category is worst"""
        mean_losses = np.mean(self.history, axis=1)
        """Sort lowest to highest"""
        order_indices = np.argsort(mean_losses)
        """Create weights as in Dr. Hasegawa-Johnson's slides (weight is number of classes performing better)
           We add one to each so that every class has some weight in the loss"""
        weights = np.linspace(1, mean_losses.shape[0], mean_losses.shape[0])
        """Assign the weights according to the proper order"""
        class_weights = {}
        for index, i in enumerate(order_indices):
            class_weights[i] = weights[index]
        """Now grab the correct weight for each speaker"""
        loss_weights = []
        for speaker in speakers:
            loss_weights.append(class_weights[self.s2i[speaker]])
        if type == 'fair':
            """Add in the lambda weighting for fair and unfair training"""
            unfair_weights = np.ones(shape=(len(loss_weights, )))
            loss_weights = np.asarray(loss_weights)

            """Lambda part"""
            loss_weights = config.train.fairness_lambda * loss_weights + (1-config.train.fairness_lambda) * unfair_weights

        elif type == 'unfair':
            """All class losses are weighted evenly, unfair"""
            loss_weights = np.ones(shape=(len(loss_weights,)))

        loss_weights = torch.from_numpy(loss_weights)
        loss_weights = self.fix_tensor(loss_weights)
        return loss_weights

    def speaker2index_and_index2speaker(self):
        self.s2i = {}
        self.i2s = {}
        for i, speaker in enumerate(list(set(config.data.speakers) - set(config.data.ignore_speakers))):
            self.s2i[speaker] = i
            self.i2s[i] = speaker

    def train(self):
        """Create speaker2index and index2speaker"""
        self.speaker2index_and_index2speaker()
        """Initialize history matrix"""
        self.history = np.random.normal(loc=0, scale=0.1, size=(len(self.s2i), config.train.class_history))
        """"""
        """"""
        iterations = 0
        if RESUME_TRAINING:
            iterations = 170000
        """Get train/test"""
        if WORDSPLIT:
            train, test = self.get_train_test_wordsplit()
        elif UTTERANCE_SPLIT:
            train, val = self.get_train_test_utterance_split()
        wordlist = joblib.load('wordlist.pkl')
        dictionary = joblib.load('dict.pkl')
        phones = joblib.load('phones.pkl')
        metadata_help = {'wordlist': wordlist, 'dictionary': dictionary, 'phones': phones}
        p2c = utils.phone2class(phones)
        c2p = utils.class2phone(phones)
        """CTC loss"""
        self.ctc_loss = nn.CTCLoss(blank=p2c[config.data.PAD_token], reduction='mean')
        # self.ctc_loss = nn.CTCLoss(blank=p2c[config.data.PAD_token], reduction='none')
        for epoch in range(config.train.num_epochs):
            """Make dataloader"""
            train_data = Dataset({'files': train, 'mode': 'train', 'metadata_help': metadata_help})
            train_gen = data.DataLoader(train_data, batch_size=config.train.batch_size,
                                        shuffle=True, collate_fn=train_data.collate, drop_last=True)
            val_data = Dataset({'files': val, 'mode': 'train', 'metadata_help': metadata_help})
            val_gen = data.DataLoader(val_data, batch_size=config.train.batch_size,
                                        shuffle=True, collate_fn=val_data.collate, drop_last=True)

            for batch_number, features in enumerate(train_gen):
                spectrograms = features['spectrograms']
                phones = features['phones']
                input_lengths = features['input_lengths']
                target_lengths = features['target_lengths']
                metadata = features["metadata"]
                # batch_speakers = [x['speaker'] for x in metadata]
                self.G = self.G.train()

                """Make input_lengths and target_lengths torch ints"""
                input_lengths = input_lengths.to(torch.int32)
                target_lengths = target_lengths.to(torch.int32)
                phones = phones.to(torch.int32)

                spectrograms = spectrograms.to(self.torch_type)

                outputs = self.G(spectrograms)

                outputs = outputs.permute(1, 0, 2)  # swap batch and sequence length dimension for CTC loss

                loss = self.ctc_loss(log_probs=outputs, targets=phones,
                                     input_lengths=input_lengths, target_lengths=target_lengths)

                """Update the loss history"""
                # self.update_history(loss, batch_speakers)
                # if epoch >= config.train.regular_epochs:
                #     loss_weights = self.get_loss_weights(batch_speakers, type='fair')
                # else:
                #     loss_weights = self.get_loss_weights(batch_speakers, type='unfair')
                # loss = loss * loss_weights

                # Backward and optimize.
                self.reset_grad()
                loss.backward()
                # loss.sum().backward()
                self.g_optimizer.step()

                if iterations % self.log_step == 0:
                    print(str(iterations) + ', loss: ' + str(loss.item()))
                    if self.use_tensorboard:
                        self.logger.scalar_summary('loss', loss.item(), iterations)

                if iterations % self.model_save_step == 0:
                    """Calculate validation loss"""
                    val_loss = self.val_loss(val=val_gen, iterations=iterations)
                    print(str(iterations) + ', val_loss: ' + str(val_loss))
                    if self.use_tensorboard:
                        self.logger.scalar_summary('val_loss', val_loss, iterations)
                """Save model checkpoints."""
                if iterations % self.model_save_step == 0:
                    G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(iterations))
                    torch.save({'model': self.G.state_dict(),
                                'optimizer': self.g_optimizer.state_dict()}, G_path)
                    print('Saved model checkpoints into {}...'.format(self.model_save_dir))

                iterations += 1

    def eval(self):
        """Evaluate trained model on test set"""
        if WORDSPLIT:
            train, test = self.get_train_test_wordsplit()
        elif UTTERANCE_SPLIT:
            test = self.get_true_test_utterances()
        wordlist = joblib.load('wordlist.pkl')
        dictionary = joblib.load('dict.pkl')
        phones = joblib.load('phones.pkl')
        metadata_help = {'wordlist': wordlist, 'dictionary': dictionary, 'phones': phones}
        p2c = utils.phone2class(phones)
        c2p = utils.class2phone(phones)
        dictionary = joblib.load('dict.pkl')
        """Convert to integers for comparison to pick best sequence"""
        for key, value in dictionary.items():
            dictionary[key] = [p2c[x] for x in value]
        stop = None
        """Get test generator"""
        test_data = Dataset({'files': test, 'mode': 'eval', 'metadata_help': metadata_help})
        test_gen = data.DataLoader(test_data, batch_size=1,
                                    shuffle=True, collate_fn=test_data.collate_eval, drop_last=True)
        for batch_number, features in tqdm(enumerate(test_gen)):
            spectrograms = features['spectrograms']
            phones = features['phones']
            batch_metadata = features['metadata'][0]
            self.G = self.G.eval()

            outputs = self.G(spectrograms)
            outputs = np.squeeze(outputs.detach().cpu().numpy())
            phones = np.squeeze(phones.detach().cpu().numpy())
            phones = phones.astype(dtype=int)
            phones = [c2p[x] for x in phones]

            output_classes = np.argmax(outputs, axis=1)

            """Decode the output predictions into a phone sequence"""
            # https://stackoverflow.com/questions/38065898/how-to-remove-the-adjacent-duplicate-value-in-a-numpy-array
            duplicates_eliminated = np.asarray([k for k, g in groupby(output_classes)])
            blanks_eliminated = duplicates_eliminated[duplicates_eliminated != 0]
            new_blanks_eliminated = []
            for x in blanks_eliminated:
                if x != p2c['SOS'] and x != p2c['EOS']:
                    new_blanks_eliminated.append(x)
            # distance = 10000000000
            if LANGUAGE_MODEL:
                """You need to add language models up to the longest sequence length dude"""
                best_sequence = language_model.get_best_sequence(ctc_output=outputs, language_models=self.language_models,
                                                                 c2p=c2p, p2c=p2c, true_transcription=phones)
                # best_sequence = language_model.get_best_sequence(outputs, self.model_2gram, self.model_3gram, c2p, p2c, phones)
                # """Write the Levenshtein distance code here"""
                # predicted_string = [chr(x) for x in new_blanks_eliminated]
                # predicted_string = ''.join(predicted_string)
                # # print(predicted_string)
                # correct_sequence = []
                # for key, value in dictionary.items():
                #     value_string = [chr(x) for x in value]
                #     value_string = ''.join(value_string)
                #     new_distance = levenshtein_distance(predicted_string, value_string)
                #     if new_distance < distance:
                #         distance = new_distance
                #         correct_sequence = value
                # stop = None




            if not LANGUAGE_MODEL:
                predicted_phones_ = [c2p[x] for x in blanks_eliminated]
            # predicted_phones_ = [c2p[x] for x in correct_sequence]
            else:
                predicted_phones_ = best_sequence
            """remove SOS and EOS"""
            predicted_phones = []
            for x in predicted_phones_:
                if x != 'SOS' and x != 'EOS':
                    predicted_phones.append(x)



            data_to_save = {'speaker': batch_metadata['speaker'],
                            'word': batch_metadata['word'],
                            'true_phones': batch_metadata['phones'],
                            'predicted_phones': predicted_phones}
            dump_path = os.path.join(self.predict_dir, batch_metadata['utterance'] + '.pkl')
            joblib.dump(data_to_save, dump_path)

    def tune_language_model_get_ctc_predictions(self):
        """Evaluate trained model on test set"""
        if WORDSPLIT:
            train, test = self.get_train_test_wordsplit()
        elif UTTERANCE_SPLIT:
            train, val = self.get_train_test_utterance_split()

        """Use subset of validation"""
        random.shuffle(val)
        # val = val[0:1000]
        val = val[0:1000]

        wordlist = joblib.load('wordlist.pkl')
        dictionary = joblib.load('dict.pkl')
        phones = joblib.load('phones.pkl')
        metadata_help = {'wordlist': wordlist, 'dictionary': dictionary, 'phones': phones}
        p2c = utils.phone2class(phones)
        c2p = utils.class2phone(phones)
        dictionary = joblib.load('dict.pkl')
        """Convert to integers for comparison to pick best sequence"""
        for key, value in dictionary.items():
            dictionary[key] = [p2c[x] for x in value]
        stop = None
        """Get test generator"""
        test_data = Dataset({'files': val, 'mode': 'eval', 'metadata_help': metadata_help})
        test_gen = data.DataLoader(test_data, batch_size=1,
                                   shuffle=True, collate_fn=test_data.collate_eval, drop_last=True)

        # K = config.language_model.K
        # CTC_weight = config.language_model.CTC_weight
        # LM_weight = config.language_model.LM_weight
        # ctc_pred_dir = 'ctc_output_predictions_tune_K_' + str(K) + '_CTC_weight_' + str(CTC_weight) + '_LM_weight_' + str(LM_weight)
        ctc_pred_dir = 'ctc_output_predictions'
        if not os.path.isdir(os.path.join(exp_dir, ctc_pred_dir)):
            os.mkdir(os.path.join(exp_dir, ctc_pred_dir))
        for batch_number, features in tqdm(enumerate(test_gen)):
            spectrograms = features['spectrograms']
            phones = features['phones']
            batch_metadata = features['metadata'][0]
            self.G = self.G.eval()

            spectrograms = spectrograms.to(torch.float32)

            outputs = self.G(spectrograms)
            outputs = np.squeeze(outputs.detach().cpu().numpy())
            phones = np.squeeze(phones.detach().cpu().numpy())
            phones = phones.astype(dtype=int)
            phones = [c2p[x] for x in phones]

            data_to_save = {'speaker': batch_metadata['speaker'],
                            'word': batch_metadata['word'],
                            'true_phones': batch_metadata['phones'],
                            'ctc_outputs': outputs}
            dump_path = os.path.join(exp_dir, ctc_pred_dir, batch_metadata['utterance'] + '.pkl')
            joblib.dump(data_to_save, dump_path)

            # output_classes = np.argmax(outputs, axis=1)

    def tune_language_model_predictions_from_ctc_output(self):
        CTC_weight = 1
        # for LM_weight in [0.1, 0.01, 0.001, 0.0001]:
        #     for K in [5, 10]:
        # for LM_weight in [0.01, 0.001, 0.0001]:
        for LM_weight in [1]:
            for K in [20]:
                lm_pred_dir = 'lm_model_predictions_tune_K_' + str(K) + '_CTC_weight_' + str(CTC_weight) + '_LM_weight_' + str(LM_weight)
                ctc_pred_dir = 'ctc_output_predictions'
                if not os.path.isdir(os.path.join(exp_dir, lm_pred_dir)):
                    os.mkdir(os.path.join(exp_dir, lm_pred_dir))
                wordlist = joblib.load('wordlist.pkl')
                dictionary = joblib.load('dict.pkl')
                phones = joblib.load('phones.pkl')
                p2c = utils.phone2class(phones)
                c2p = utils.class2phone(phones)
                files_to_evaluate = collect_files(os.path.join(exp_dir, ctc_pred_dir))

                for file in tqdm(files_to_evaluate):
                    filename = file.split('/')[-1]
                    data = joblib.load(file)
                    outputs = data['ctc_outputs']

                    """You need to add language models up to the longest sequence length dude"""
                    best_sequence = language_model.get_best_sequence(ctc_output=outputs,
                                                                     language_models=self.language_models,
                                                                     c2p=c2p, p2c=p2c, true_transcription=data['true_phones'],
                                                                     K=K, CTC_weight=CTC_weight, LM_weight=LM_weight)


                    predicted_phones_ = best_sequence
                    """remove SOS and EOS"""
                    predicted_phones = []
                    for x in predicted_phones_:
                        if x != 'SOS' and x != 'EOS':
                            predicted_phones.append(x)

                    data_to_save = {'speaker': data['speaker'],
                                    'word': data['word'],
                                    'true_phones': data['true_phones'],
                                    'predicted_phones': predicted_phones}
                    dump_path = os.path.join(exp_dir, lm_pred_dir, filename)
                    joblib.dump(data_to_save, dump_path)

    def tune_language_model_performance(self):
        """Set the hyperparameters"""
        """"""
        if not os.path.isdir(os.path.join(exp_dir, 'LM_tuning_performance')):
            os.mkdir(os.path.join(exp_dir, 'LM_tuning_performance'))
        CTC_weight = 1
        # for LM_weight in [0.1, 0.01, 0.001, 0.0001]:
        #     for K in [5, 10]:
        # for LM_weight in [0.01, 0.001, 0.0001]:
        for LM_weight in [1]:
            for K in [20]:
                speakers = {}
                #for file in tqdm(collect_files(self.predict_dir)):
                lm_pred_dir = 'lm_model_predictions_tune_K_' + str(K) + '_CTC_weight_' + str(CTC_weight) + '_LM_weight_' + str(LM_weight)
                for file in tqdm(collect_files(os.path.join(exp_dir, lm_pred_dir))):
                    utterance = file.split('/')[-1]
                    if ATTENTION or DCGAN:
                        speaker = utterance.split('_')[7]  # the speaker we're converting to
                    else:
                        speaker = utterance.split('_')[0]  # the speaker we're converting to
                    if speaker not in speakers:
                        data = joblib.load(file)
                        data['filename'] = file.split('/')[-1]
                        speakers[speaker] = [data]
                    else:
                        data = joblib.load(file)
                        data['filename'] = file.split('/')[-1]
                        speakers[speaker].append(data)

                delimiter = '_'
                """Percent correct words"""
                correct_words = {}
                WER = {}
                for speaker, utt_list in speakers.items():
                    correct_word_count = 0
                    for i, utt in enumerate(utt_list):
                        true_seq = delimiter.join(utt['true_phones'])
                        pred_seq = delimiter.join(utt['predicted_phones'])
                        if true_seq == pred_seq:
                            correct_word_count += 1
                    word_accuracy = correct_word_count/(i+1)
                    correct_words[speaker] = word_accuracy
                    WER[speaker] = (1 - word_accuracy) * 100

                """Percent phones recognized in the utterance that are in the true phones"""
                delimiter = '_'
                correct_perc = {}
                for speaker, utt_list in speakers.items():
                    correct_word_perc = 0
                    for i, utt in enumerate(utt_list):
                        true_phones = set(utt['true_phones'])
                        pred_phones = set(utt['predicted_phones'])
                        intersection = true_phones.intersection(pred_phones)
                        percent_correct_phones = len(list(intersection))/len(list(true_phones))
                        correct_word_perc += percent_correct_phones
                    word_accuracy = correct_word_perc / (i + 1)
                    correct_perc[speaker] = word_accuracy

                hyperparams_prefix = '_K_' + str(K) + '_CTC_weight_' + str(CTC_weight) + '_LM_weight_' + str(LM_weight)
                self.dump_json(dict=correct_words, path=os.path.join(exp_dir, 'LM_tuning_performance', 'test_accuracies' + hyperparams_prefix + '.json'))
                """Let's sort best to worst WER"""
                sorted_WER = {k: v for k, v in sorted(WER.items(), key=lambda item: item[1])}
                if not UNSEEN_NORMAL_BASELINE:
                    self.dump_json(dict=sorted_WER, path=os.path.join(exp_dir, 'LM_tuning_performance', 'test_WER' + hyperparams_prefix + '_MEAN_' + str(np.mean(np.asarray([value for key, value in WER.items() if 'C' not in key]))) + '.json'))
                else:
                    self.dump_json(dict=sorted_WER, path=os.path.join(exp_dir, 'LM_tuning_performance',
                                                                      'test_WER' + hyperparams_prefix + '_MEAN_' + str(
                                                                          np.mean(np.asarray(
                                                                              [value for key, value in WER.items()]))) + '.json'))


                stats = {}
                """Add more code for different stats on the results here"""
                """Let's compare dysarthric performance to normal performance"""
                stats['dysarthric_mean_WER'] = np.mean(np.asarray([value for key, value in WER.items() if 'C' not in key]))
                stats['normal_mean_WER'] = np.mean(np.asarray([value for key, value in WER.items() if 'C' in key]))
                stats['female_WER'] = np.mean(np.asarray([value for key, value in WER.items() if 'F' in key]))
                stats['male_WER'] = np.mean(np.asarray([value for key, value in WER.items() if 'M' in key]))
                self.dump_json(dict=stats, path=os.path.join(exp_dir, 'LM_tuning_performance', 'test_stats' + hyperparams_prefix + '.json'))


    def eval_language_model_ctc_predictions(self):
        """Evaluate trained model on test set"""
        if WORDSPLIT:
            train, test = self.get_train_test_wordsplit()
        elif UTTERANCE_SPLIT:
            test = self.get_true_test_utterances()
        wordlist = joblib.load('wordlist.pkl')
        dictionary = joblib.load('dict.pkl')
        phones = joblib.load('phones.pkl')
        metadata_help = {'wordlist': wordlist, 'dictionary': dictionary, 'phones': phones}
        p2c = utils.phone2class(phones)
        c2p = utils.class2phone(phones)
        dictionary = joblib.load('dict.pkl')
        """Convert to integers for comparison to pick best sequence"""
        for key, value in dictionary.items():
            dictionary[key] = [p2c[x] for x in value]
        stop = None
        """Get test generator"""
        test_data = Dataset({'files': test, 'mode': 'eval', 'metadata_help': metadata_help})
        test_gen = data.DataLoader(test_data, batch_size=1,
                                   shuffle=True, collate_fn=test_data.collate_eval, drop_last=True)
        if not os.path.isdir(os.path.join(exp_dir, 'ctc_output_predictions_test')):
            os.mkdir(os.path.join(exp_dir, 'ctc_output_predictions_test'))
        for batch_number, features in tqdm(enumerate(test_gen)):
            spectrograms = features['spectrograms']
            phones = features['phones']
            batch_metadata = features['metadata'][0]
            self.G = self.G.eval()

            outputs = self.G(spectrograms)
            outputs = np.squeeze(outputs.detach().cpu().numpy())
            phones = np.squeeze(phones.detach().cpu().numpy())
            phones = phones.astype(dtype=int)
            phones = [c2p[x] for x in phones]

            data_to_save = {'speaker': batch_metadata['speaker'],
                            'word': batch_metadata['word'],
                            'true_phones': batch_metadata['phones'],
                            'ctc_outputs': outputs}
            dump_path = os.path.join(exp_dir, 'ctc_output_predictions_test', batch_metadata['utterance'] + '.pkl')
            joblib.dump(data_to_save, dump_path)

            # output_classes = np.argmax(outputs, axis=1)

    def language_model_predictions_from_ctc_output(self):
        """Set the hyperparameters"""
        if ATTENTION:
            K = config.language_model.Attention_K
            CTC_weight = config.language_model.Attention_CTC_weight
            LM_weight = config.language_model.Attention_LM_weight
        if DCGAN:
            K = config.language_model.DCGAN_K
            CTC_weight = config.language_model.DCGAN_CTC_weight
            LM_weight = config.language_model.DCGAN_LM_weight
        if ORIGINAL_UNSEEN_BASELINE:
            K = config.language_model.Oracle_K
            CTC_weight = config.language_model.Oracle_CTC_weight
            LM_weight = config.language_model.Oracle_LM_weight
        if BASELINE:
            K = config.language_model.Limited_K
            CTC_weight = config.language_model.Limited_CTC_weight
            LM_weight = config.language_model.Limited_LM_weight
        if UNSEEN_NORMAL_BASELINE:
            K = config.language_model.Lack_K
            CTC_weight = config.language_model.Lack_CTC_weight
            LM_weight = config.language_model.Lack_LM_weight
        if not os.path.isdir(os.path.join(exp_dir, 'language_model_predictions_test')):
            os.mkdir(os.path.join(exp_dir, 'language_model_predictions_test'))
        wordlist = joblib.load('wordlist.pkl')
        dictionary = joblib.load('dict.pkl')
        phones = joblib.load('phones.pkl')
        p2c = utils.phone2class(phones)
        c2p = utils.class2phone(phones)
        files_to_evaluate = collect_files(os.path.join(exp_dir, 'ctc_output_predictions_test'))
        # multiprocessing_data = []
        # for file in files_to_evaluate:
        #     datum = {'file': file,
        #              'lm': self.language_models,
        #              'c2p': c2p,
        #              'p2c': p2c}
        #     multiprocessing_data.append(datum)
        # # multiprocessing_lm_eval(multiprocessing_data[0])
        # with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        #     for _ in tqdm(executor.map(multiprocessing_lm_eval, multiprocessing_data)):
        #         """"""
        for file in tqdm(files_to_evaluate):
            filename = file.split('/')[-1]
            data = joblib.load(file)
            outputs = data['ctc_outputs']

            """You need to add language models up to the longest sequence length dude"""
            best_sequence = language_model.get_best_sequence(ctc_output=outputs,
                                                             language_models=self.language_models,
                                                             c2p=c2p, p2c=p2c, true_transcription=phones,
                                                             K=K, CTC_weight=CTC_weight, LM_weight=LM_weight)


            predicted_phones_ = best_sequence
            """remove SOS and EOS"""
            predicted_phones = []
            for x in predicted_phones_:
                if x != 'SOS' and x != 'EOS':
                    predicted_phones.append(x)

            data_to_save = {'speaker': data['speaker'],
                            'word': data['word'],
                            'true_phones': data['true_phones'],
                            'predicted_phones': predicted_phones}
            dump_path = os.path.join(exp_dir, 'language_model_predictions_test', filename)
            joblib.dump(data_to_save, dump_path)



    def performance(self):
        """"""
        speakers = {}
        for file in tqdm(collect_files(self.predict_dir)):
        #for file in tqdm(collect_files(os.path.join(exp_dir, 'language_model_predictions'))):
            utterance = file.split('/')[-1]
            speaker = utterance.split('_')[0]
            if speaker not in speakers:
                data = joblib.load(file)
                data['filename'] = file.split('/')[-1]
                speakers[speaker] = [data]
            else:
                data = joblib.load(file)
                data['filename'] = file.split('/')[-1]
                speakers[speaker].append(data)

        delimiter = '_'
        """Percent correct words"""
        correct_words = {}
        WER = {}
        for speaker, utt_list in speakers.items():
            correct_word_count = 0
            for i, utt in enumerate(utt_list):
                true_seq = delimiter.join(utt['true_phones'])
                pred_seq = delimiter.join(utt['predicted_phones'])
                if true_seq == pred_seq:
                    correct_word_count += 1
                else:
                    print('True seq: ' + true_seq)
                    print('Pred seq: ' + pred_seq)
            word_accuracy = correct_word_count/(i+1)
            correct_words[speaker] = word_accuracy
            WER[speaker] = (1 - word_accuracy) * 100

        """Percent phones recognized in the utterance that are in the true phones"""
        delimiter = '_'
        correct_perc = {}
        for speaker, utt_list in speakers.items():
            correct_word_perc = 0
            for i, utt in enumerate(utt_list):
                true_phones = set(utt['true_phones'])
                pred_phones = set(utt['predicted_phones'])
                intersection = true_phones.intersection(pred_phones)
                percent_correct_phones = len(list(intersection))/len(list(true_phones))
                correct_word_perc += percent_correct_phones
            word_accuracy = correct_word_perc / (i + 1)
            correct_perc[speaker] = word_accuracy

        if not LANGUAGE_MODEL:
            self.dump_json(dict=correct_words, path=os.path.join(exp_dir, 'test_accuracies_no_LM.json'))
        else:
            self.dump_json(dict=correct_words, path=os.path.join(exp_dir, 'test_accuracies_LM.json'))
        """Let's sort best to worst WER"""
        sorted_WER = {k: v for k, v in sorted(WER.items(), key=lambda item: item[1])}
        if not LANGUAGE_MODEL:
            self.dump_json(dict=sorted_WER, path=os.path.join(exp_dir, 'test_WER_no_LM.json'))
        else:
            self.dump_json(dict=sorted_WER, path=os.path.join(exp_dir, 'test_WER_LM.json'))

        stats = {}
        """Add more code for different stats on the results here"""
        """Let's compare dysarthric performance to normal performance"""
        stats['dysarthric_mean_WER'] = np.mean(np.asarray([value for key, value in WER.items() if 'C' not in key]))
        stats['normal_mean_WER'] = np.mean(np.asarray([value for key, value in WER.items() if 'C' in key]))
        stats['female_WER'] = np.mean(np.asarray([value for key, value in WER.items() if 'F' in key]))
        stats['male_WER'] = np.mean(np.asarray([value for key, value in WER.items() if 'M' in key]))
        if not LANGUAGE_MODEL:
            self.dump_json(dict=stats, path=os.path.join(exp_dir, 'test_stats_no_LM.json'))
        else:
            self.dump_json(dict=stats, path=os.path.join(exp_dir, 'test_stats_LM.json'))


    def to_gpu(self, tensor):
        tensor = tensor.to(self.torch_type)
        tensor = tensor.to(self.device)
        return tensor

    def fix_tensor(self, x):
        x.requires_grad = True
        x = x.to(self.torch_type)
        x = x.cuda()
        return x

    def dump_json(self, dict, path):
        a_file = open(path, "w")
        json.dump(dict, a_file, indent=2)
        a_file.close()

def main():
    solver = Solver()
    if TRAIN:
        solver.train()
    if TUNE_LANGUAGE_MODEL:
        solver.tune_language_model_get_ctc_predictions()
        solver.tune_language_model_predictions_from_ctc_output()
        solver.tune_language_model_performance()
    if EVAL:
        if LANGUAGE_MODEL:
            solver.eval_language_model_ctc_predictions()
            solver.language_model_predictions_from_ctc_output()
        else:
            solver.eval()
    if PERFORMANCE_EVAL:
        solver.performance()


if __name__ == "__main__":
    main()