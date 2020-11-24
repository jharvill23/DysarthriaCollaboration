"""
Used for various functions and the creation of:
dict.pkl
wordlist.pkl
phones.pkl
partition.pkl
"""

import os
import pandas as pd
import joblib
from preprocessing import collect_files
import shutil
from tqdm import tqdm
import yaml
from easydict import EasyDict as edict
import random

config = edict(yaml.load(open('config.yml'), Loader=yaml.SafeLoader))

def get_wordlist(wordlist='wordlist.csv'):
    """Read file"""
    wordlist = pd.read_csv(wordlist)
    """Convert to dictionary"""
    wordlist = wordlist.to_dict('r')
    new_wordlist = {}
    for pair in wordlist:
        new_wordlist[pair['FILE NAME']] = pair['WORD']
    wordlist = new_wordlist

    """If the key doesn't contain an underscore, it has B1, B2, and B3
    so we need to make three new keys and delete the old one"""
    keys_to_change = []
    for key, value in wordlist.items():
        if '_' not in key:
            keys_to_change.append(key)
    for key in keys_to_change:
        value = wordlist[key]
        wordlist['B1_' + key] = value
        wordlist['B2_' + key] = value
        wordlist['B3_' + key] = value
        del wordlist[key]

    joblib.dump(wordlist, 'wordlist.pkl')

def get_dictionary(dict='dict.txt'):
    """"""
    new_dict = {}
    with open(dict) as f:
        for l in f:
            l = l.replace('\n', '')
            word = l.split(' ')[0]
            phones = l.split(' ')[1:]
            new_dict[word] = phones
    joblib.dump(new_dict, 'dict.pkl')

def get_phones():
    dictionary = joblib.load('dict.pkl')
    phones = []
    for key, value in dictionary.items():
        phones.extend(value)
    phones = list(set(phones))

    """Add in the PAD, SOS, and EOS tokens"""
    phones.append(config.data.PAD_token)
    phones.append(config.data.SOS_token)
    phones.append(config.data.EOS_token)

    """Write to disk"""
    joblib.dump(phones, 'phones.pkl')

def get_seen_unseen_words():
    dictionary = joblib.load('dict.pkl')
    words = [key for key, value in dictionary.items()]
    random.shuffle(words)
    num_seen_words = round(len(words)*config.data.seen_unseen_split)
    seen_words = words[0:num_seen_words]
    unseen_words = words[num_seen_words:]
    # intersection = set(seen_words).intersection(set(unseen_words))
    seen_unseen_1 = {'seen': seen_words, 'unseen': unseen_words}
    seen_unseen_2 = {'seen': unseen_words, 'unseen': seen_words}  # we flip for the other partition!
    seen_unseen = {'partition_1': seen_unseen_1, 'partition_2': seen_unseen_2}

    # # check the partitions
    # part_1_intersect = set(seen_unseen_1['seen']).intersection(set(seen_unseen_1['unseen']))
    # part_2_intersect = set(seen_unseen_2['seen']).intersection(set(seen_unseen_2['unseen']))

    joblib.dump(seen_unseen, 'seen_unseen_partitions.pkl')

def get_partition_wordsplit():
    wordlist = joblib.load(('wordlist.pkl'))
    unique_words = []
    [unique_words.append(value) for key, value in wordlist.items()]
    unique_words = list(set(unique_words))
    random.shuffle(unique_words)
    split_index = int(config.data.train_test_frac*len(unique_words))
    train = unique_words[0:split_index]
    test = unique_words[split_index:]
    # overlap = (set(train)).intersection(set(test))
    """Dump to disk"""
    joblib.dump({'train': train, 'test': test}, 'partition.pkl')

def get_file_metadata(file, wordlist=None, dictionary=None):
    """Return speaker, word, and phones of given file
    If wordlist or dictionary aren't provided, load them from file"""
    if wordlist == None:
        wordlist = joblib.load('wordlist.pkl')
    if dictionary == None:
        dictionary = joblib.load('dict.pkl')
    utterance = file.split('/')[-1]
    speaker = utterance.split('_')[0]
    delimiter = '_'
    word = wordlist[delimiter.join((utterance.split('_')[1], utterance.split('_')[2]))]
    phones = dictionary[word.upper()]
    return {'speaker': speaker, 'word': word, 'phones': phones, 'utterance': utterance[:-4]}

def get_file_metadata_vc(file, wordlist=None, dictionary=None):
    """Return speaker, word, and phones of given file
    If wordlist or dictionary aren't provided, load them from file"""
    if wordlist == None:
        wordlist = joblib.load('wordlist.pkl')
    if dictionary == None:
        dictionary = joblib.load('dict.pkl')
    utterance = file.split('/')[-1]
    normal_speaker = utterance.split('_')[0]
    dys_speaker = utterance.split('_')[2]
    delimiter = '_'
    word_argument = delimiter.join(utterance.split('_')[3:5])
    normal_filename = (utterance.split('_')[0] + delimiter + delimiter.join(utterance.split('_')[3:6]))[:-4]
    word = wordlist[word_argument]
    phones = dictionary[word.upper()]
    return {'normal_speaker': normal_speaker, 'dys_speaker': dys_speaker, 'word': word,
            'phones': phones, 'utterance': utterance[:-4], 'normal_filename': normal_filename}

def get_file_metadata_ctc_training(file, wordlist=None, dictionary=None):
    """Return speaker, word, and phones of given file
        If wordlist or dictionary aren't provided, load them from file"""
    if wordlist == None:
        wordlist = joblib.load('wordlist.pkl')
    if dictionary == None:
        dictionary = joblib.load('dict.pkl')
    utterance = file.split('/')[-1]
    normal_speaker = utterance.split('_')[5]
    dys_speaker = utterance.split('_')[0]
    delimiter = '_'
    word_argument = delimiter.join(utterance.split('_')[1:3])
    # normal_filename = (utterance.split('_')[0] + delimiter + delimiter.join(utterance.split('_')[3:6]))[:-4]
    word = wordlist[word_argument]
    phones = dictionary[word.upper()]
    return {'normal_speaker': normal_speaker, 'dys_speaker': dys_speaker, 'word': word,
            'phones': phones, 'utterance': utterance[:-4]}

def get_vc_partition_utterance_split():
    """Set aside one utterance for each speaker of each word for test set and val set"""
    wordlist = joblib.load('wordlist.pkl')
    dictionary = joblib.load('dict.pkl')
    seen_unseen_words = joblib.load('seen_unseen_words.pkl')
    # unique_words = set([value for key, value in wordlist.items()])
    speaker_test_files = {}
    speaker_val_files = {}
    test_files = []
    val_files = []
    all_files = collect_files(config.directories.features)
    new_all_files = []
    for file in all_files:
        metadata = get_file_metadata(file, wordlist=wordlist, dictionary=dictionary)
        word = (metadata['word']).upper()
        if word in seen_unseen_words['seen']:
            new_all_files.append(file)
            if metadata["speaker"] not in speaker_test_files:
                speaker_test_files[metadata["speaker"]] = {}
            if metadata["speaker"] not in speaker_val_files:
                speaker_val_files[metadata["speaker"]] = {}
            if metadata["word"] not in speaker_test_files[metadata["speaker"]]:
                speaker_test_files[metadata["speaker"]][metadata["word"]] = file
                test_files.append(file)
            if metadata["word"] in speaker_test_files[metadata["speaker"]] \
                    and metadata["word"] not in speaker_val_files[metadata["speaker"]] \
                    and speaker_test_files[metadata["speaker"]][metadata["word"]] != file:
                speaker_val_files[metadata["speaker"]][metadata["word"]] = file
                val_files.append(file)
    """All files stored in test_files is the test set
       All files stored in val_files is val set
       The list of all other files is the train set"""

    """Let's check that there's no intersection between test and val sets"""
    # test_val_intersection = list(set(test_files).intersection(set(val_files)))
    train_files = list(set(new_all_files) - set(test_files) - set(val_files))
    intersection_test = (set(train_files)).intersection(set(test_files))  # check there are no overlapping elements
    intersection_val = (set(train_files)).intersection(set(val_files))

    """Check that none of the unseen words are in any of these partitions"""
    # seen_files = train_files + test_files + val_files
    # for file in seen_files:
    #     metadata = get_file_metadata(file, wordlist=wordlist, dictionary=dictionary)
    #     word = (metadata['word']).upper()
    #     if word in seen_unseen_words['unseen']:
    #         print("unseen word in seen partition, something is wrong with code")
    # seen_files = list((set(train_files).add(set(test_files))).add(set(val_files)))
    joblib.dump({'train': train_files, 'test': test_files, 'val': val_files}, 'vc_partition.pkl')

def get_full_filenames_data_partition_utterance_split():
    """Set aside one utterance for each speaker of each word for test set and val set"""
    wordlist = joblib.load('wordlist.pkl')
    dictionary = joblib.load('dict.pkl')
    seen_unseen_partitions = joblib.load('seen_unseen_partitions.pkl')

    full_partition = {}

    print('Creating full partition...')

    for key, partition in seen_unseen_partitions.items():
        seen_unseen_words = partition
        big_partition = {}
        """Now we want to create train/val/test partitions for both seen and unseen for every speaker"""
        for partition_type in ['seen', 'unseen']:
            speaker_test_files = {}
            speaker_val_files = {}
            speaker_train_files = {}
            test_files = []
            val_files = []
            train_files = []
            all_files = collect_files(config.directories.features)
            new_all_files = []
            for file in all_files:
                metadata = get_file_metadata(file, wordlist=wordlist, dictionary=dictionary)
                word = (metadata['word']).upper()
                if word in seen_unseen_words[partition_type]:
                    new_all_files.append(file)
                    if metadata["speaker"] not in speaker_test_files:
                        speaker_test_files[metadata["speaker"]] = {}
                    if metadata["speaker"] not in speaker_val_files:
                        speaker_val_files[metadata["speaker"]] = {}
                    if metadata["speaker"] not in speaker_train_files:
                        speaker_train_files[metadata["speaker"]] = {}
                    if metadata["word"] not in speaker_test_files[metadata["speaker"]]:
                        speaker_test_files[metadata["speaker"]][metadata["word"]] = file
                        test_files.append(file)
                    if metadata["word"] in speaker_test_files[metadata["speaker"]] \
                            and metadata["word"] not in speaker_val_files[metadata["speaker"]] \
                            and speaker_test_files[metadata["speaker"]][metadata["word"]] != file:
                        speaker_val_files[metadata["speaker"]][metadata["word"]] = file
                        val_files.append(file)
                    if metadata["word"] in speaker_test_files[metadata["speaker"]] \
                        and metadata["word"] in speaker_val_files[metadata["speaker"]] \
                        and metadata["word"] not in speaker_train_files[metadata["speaker"]] \
                        and speaker_test_files[metadata["speaker"]][metadata["word"]] != file \
                        and speaker_val_files[metadata["speaker"]][metadata["word"]] != file:
                        speaker_train_files[metadata["speaker"]][metadata["word"]] = [file]
                        train_files.append(file)
                    elif metadata["word"] in speaker_test_files[metadata["speaker"]] \
                        and metadata["word"] in speaker_val_files[metadata["speaker"]] \
                        and speaker_test_files[metadata["speaker"]][metadata["word"]] != file \
                        and speaker_val_files[metadata["speaker"]][metadata["word"]] != file:
                        speaker_train_files[metadata["speaker"]][metadata["word"]].append(file)
                        train_files.append(file)
            """All files stored in test_files is the test set
               All files stored in val_files is val set
               The list of all other files is the train set"""

            """Now we have files sorted by speaker into train/val/test sets"""
            """Let's put them into normal and dysarthric speaker sets"""

            big_partition[partition_type] = {'normal': {'train': {}, 'val': {}, 'test': {}},
                                             'dys': {'train': {}, 'val': {}, 'test': {}},
                                             'filenames_only': {'normal': {'train': {}, 'val': {}, 'test': {}},
                                             'dys': {'train': {}, 'val': {}, 'test': {}}}}

            for speaker, value in speaker_test_files.items():
                if "C" not in speaker:
                    if speaker in config.data.conversion_target_speakers:
                        big_partition[partition_type]['dys']['test'][speaker] = value
                else:
                    if speaker in config.data.conversion_source_speakers:
                        big_partition[partition_type]['normal']['test'][speaker] = value

            for speaker, value in speaker_train_files.items():
                if "C" not in speaker:
                    if speaker in config.data.conversion_target_speakers:
                        big_partition[partition_type]['dys']['train'][speaker] = value
                else:
                    if speaker in config.data.conversion_source_speakers:
                        big_partition[partition_type]['normal']['train'][speaker] = value

            for speaker, value in speaker_val_files.items():
                if "C" not in speaker:
                    if speaker in config.data.conversion_target_speakers:
                        big_partition[partition_type]['dys']['val'][speaker] = value
                else:
                    if speaker in config.data.conversion_source_speakers:
                        big_partition[partition_type]['normal']['val'][speaker] = value

            #big_partition[partition_type]['filenames_only'] = {}
            """We've nicely split by speaker, now we combine into just filename lists"""
            for speech_type in ['normal', 'dys']:
                for parttype in ['train', 'val', 'test']:
                    filelist = []
                    if parttype == 'val' or parttype == 'test':
                        for key_, value in big_partition[partition_type][speech_type][parttype].items():
                            #sublist = []
                            for subkey, subfile in value.items():
                                filelist.append(subfile)
                    if parttype == 'train':
                        for key_, value in big_partition[partition_type][speech_type][parttype].items():
                            #sublist = []
                            for subkey, subfiles in value.items():
                                filelist.extend(subfiles)
                    big_partition[partition_type]['filenames_only'][speech_type][parttype] = filelist
            stop = None
        full_partition[key] = big_partition
    joblib.dump(full_partition, 'full_filenames_data_partition.pkl')
    print('Done creating full partition.')
            # """Let's check that there's no intersection between test and val sets"""
            # # test_val_intersection = list(set(test_files).intersection(set(val_files)))
            # train_files_ = list(set(new_all_files) - set(test_files) - set(val_files))
            # if set(train_files_) == set(train_files):
            #     stop = None
            # intersection_test = (set(train_files)).intersection(set(test_files))  # check there are no overlapping elements
            # intersection_val = (set(train_files)).intersection(set(val_files))
            #
            # """Check that none of the unseen words are in any of these partitions"""
            # # seen_files = train_files + test_files + val_files
            # # for file in seen_files:
            # #     metadata = get_file_metadata(file, wordlist=wordlist, dictionary=dictionary)
            # #     word = (metadata['word']).upper()
            # #     if word in seen_unseen_words['unseen']:
            # #         print("unseen word in seen partition, something is wrong with code")
            # # seen_files = list((set(train_files).add(set(test_files))).add(set(val_files)))
            # joblib.dump({'train': train_files, 'test': test_files, 'val': val_files}, 'vc_partition.pkl')

def get_ctc_partition_offline_augmentation():
    """NOTE: WE DON'T HAVE A TEST PARTITION, BECAUSE WE TEST ON REAL DYSARTHRIC SPEECH, NOT AUGMENTED"""
    augmented_files = collect_files(config.directories.spec_augmented_data)
    random.shuffle(augmented_files)
    split_index = round(0.98 * len(augmented_files))
    train_files = augmented_files[0:split_index]
    val_files = augmented_files[split_index:]
    intersection = set(train_files).intersection(set(val_files))
    data = {'train': train_files, 'val': val_files}
    joblib.dump(data, 'ctc_partition.pkl')

def get_dcgan_ctc_partition_offline_augmentation():
    """NOTE: WE DON'T HAVE A TEST PARTITION, BECAUSE WE TEST ON REAL DYSARTHRIC SPEECH, NOT AUGMENTED"""
    augmented_files = collect_files(config.directories.spec_augmented_data_dcgan)
    random.shuffle(augmented_files)
    split_index = round(0.98 * len(augmented_files))
    train_files = augmented_files[0:split_index]
    val_files = augmented_files[split_index:]
    intersection = set(train_files).intersection(set(val_files))
    data = {'train': train_files, 'val': val_files}
    joblib.dump(data, 'dcgan_ctc_partition.pkl')

def get_original_unseen_ctc_partition_offline_augmentation():
    """NOTE: WE DON'T HAVE A TEST PARTITION, BECAUSE WE TEST ON REAL DYSARTHRIC SPEECH, NOT AUGMENTED"""
    augmented_files = collect_files(config.directories.spec_augmented_unseen_data)
    random.shuffle(augmented_files)
    split_index = round(0.98 * len(augmented_files))
    train_files = augmented_files[0:split_index]
    val_files = augmented_files[split_index:]
    intersection = set(train_files).intersection(set(val_files))
    data = {'train': train_files, 'val': val_files}
    joblib.dump(data, 'original_unseen_ctc_partition.pkl')

def get_unseen_normal_ctc_partition_offline_augmentation():
    """NOTE: WE DON'T HAVE A TEST PARTITION, BECAUSE WE TEST ON REAL DYSARTHRIC SPEECH, NOT AUGMENTED"""
    augmented_files = collect_files(config.directories.spec_augmented_unseen_normal_data)
    random.shuffle(augmented_files)
    split_index = round(0.98 * len(augmented_files))
    train_files = augmented_files[0:split_index]
    val_files = augmented_files[split_index:]
    intersection = set(train_files).intersection(set(val_files))
    data = {'train': train_files, 'val': val_files}
    joblib.dump(data, 'unseen_normal_ctc_partition.pkl')

def get_baseline_ctc_partition_offline_augmentation():
    """NOTE: WE DON'T HAVE A TEST PARTITION, BECAUSE WE TEST ON REAL DYSARTHRIC SPEECH, NOT AUGMENTED"""
    augmented_files = collect_files(config.directories.spec_augmented_seen_data_dysarthric)
    random.shuffle(augmented_files)
    split_index = round(0.98 * len(augmented_files))
    train_files = augmented_files[0:split_index]
    val_files = augmented_files[split_index:]
    intersection = set(train_files).intersection(set(val_files))
    data = {'train': train_files, 'val': val_files}
    joblib.dump(data, 'baseline_ctc_partition.pkl')

def get_true_test_utterances_ctc():
    all_files = collect_files(config.directories.features)
    vc_partition = joblib.load("vc_partition.pkl")
    potential_test_utterances = list(set(all_files).difference(set(vc_partition['train'] + vc_partition['test'] + vc_partition['val'])))
    """Now we only want the utterances with the dysarthic speakers"""
    test_utterances = []
    unique_dys_speakers = []
    for x in potential_test_utterances:
        utterance = x.split('/')[1]
        speaker = utterance.split('_')[0]
        if speaker in config.data.conversion_target_speakers:
            test_utterances.append(x)
            if speaker not in unique_dys_speakers:
                unique_dys_speakers.append(speaker)
    """Just do a quick sanity check that the words from these utterances aren't in the seen words"""
    # wordlist = joblib.load('wordlist.pkl')
    # dictionary = joblib.load('dict.pkl')
    # seen_unseen_words = joblib.load('seen_unseen_words.pkl')
    # for x in test_utterances:
    #     metadata = get_file_metadata(x, wordlist=wordlist, dictionary=dictionary)
    #     if metadata['word'].upper() in seen_unseen_words['seen']:
    #         print("Messed up code...")
    return test_utterances

def check_full_partition():
    full_partition = joblib.load('full_filenames_data_partition.pkl')
    separate_filename_sets = []
    for key, part in full_partition.items():
        for type in ['seen', 'unseen']:
            for speaker_type in ['normal', 'dys']:
                for parttype in ['train', 'test', 'val']:
                    little_set = set(full_partition[key][type]['filenames_only'][speaker_type][parttype])
                    separate_filename_sets.append(little_set)
        """See if any intersect"""
        for set_ in separate_filename_sets:
            for set__ in separate_filename_sets:
                if set_ != set__:
                    if set_.intersection(set__):
                        print('Overlap detected!')
                    else:
                        print('No overlap...')

if not os.path.exists('wordlist.pkl'):
    get_wordlist('wordlist.csv')

if not os.path.exists('dict.pkl'):
    get_dictionary('dict.txt')

if not os.path.exists('phones.pkl'):
    get_phones()

if not os.path.exists('seen_unseen_words.pkl'):
    get_seen_unseen_words()

if not os.path.exists('full_filenames_data_partition.pkl'):
    get_full_filenames_data_partition_utterance_split()


def phone2class(phones=None):
    if phones == None:
        phones = joblib.load('phones.pkl')
    p2c = {}
    # p2c = {config.data.PAD_token: 0, config.data.SOS_token: 1, config.data.EOS_token:2}
    for i, phone in enumerate(phones):
        p2c[phone] = i + 3
    p2c[config.data.PAD_token] = 0
    p2c[config.data.SOS_token] = 1
    p2c[config.data.EOS_token] = 2
    return p2c

def class2phone(phones=None):
    p2c = phone2class(phones)
    c2p = {}
    for key, value in p2c.items():
        c2p[value] = key
    return c2p

def dummy_check_metadata():
    """I used this to check that my assumptions about the naming convention
    are correct (B1, B2, B3 need to be added to certain words)
    After listening, the audios labeled with their words are actually those words so assumptions are correct"""
    files = collect_files('silence_removed')
    dummy_dir = 'dummy_files_check'
    if not os.path.isdir(dummy_dir):
        os.mkdir(dummy_dir)
    for i, file in tqdm(enumerate(files)):
        metadata = get_file_metadata(file)
        target_path = os.path.join(dummy_dir,
                                   metadata['speaker'] + '_' + metadata['word'] + '_' + str(i) + '.wav')
        shutil.copy(file, target_path)

def get_vocab_size():
    dictionary = joblib.load('phones.pkl')
    return len(dictionary)


def main():
    """"""
    # get_full_filenames_data_partition_utterance_split()
    # check_full_partition()
    # p2c = phone2class()
    # c2p = class2phone()
    # vocab_size = get_vocab_size()


if __name__ == '__main__':
    main()