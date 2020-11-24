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
from nltk.util import bigrams
from nltk.util import ngrams
from nltk.lm import MLE
from nltk.util import everygrams

"""I KEPT THIS JUST TO HAVE BUT THIS ALGORITHM IS INCORRECT.
   I DIDN'T ACCUMULATE CTC PROBABILITIES AND HAVE NO IDEA HOW IT EVEN REMOTELY WORKED BEFORE."""

config = edict(yaml.load(open('config.yml'), Loader=yaml.SafeLoader))

# N_GRAM = 3
# K = config.language_model.K  # beam width
# ALPHA = config.language_model.CTC_weight  # CTC probabilities weight
# BETA = config.language_model.LM_weight  # language model probabilities weight
epsilon = 10e-300
TOP_SYMBOLS = 50  # now it has no effect

def get_best_sequence(ctc_output, language_models, c2p, p2c, true_transcription, K, CTC_weight, LM_weight):
    ALPHA = CTC_weight
    BETA = LM_weight
    lm1 = language_models['model_1gram']
    lm2 = language_models['model_2gram']
    lm3 = language_models['model_3gram']
    lm4 = language_models['model_4gram']
    lm5 = language_models['model_5gram']
    lm6 = language_models['model_6gram']
    lm7 = language_models['model_7gram']
    lm8 = language_models['model_8gram']
    lm9 = language_models['model_9gram']
    lm10 = language_models['model_10gram']
    lm11 = language_models['model_11gram']
    lm12 = language_models['model_12gram']
    lm13 = language_models['model_13gram']
    correct_sequence = true_transcription
    """This seems to be close, but it's not right so check it thoroughly later"""

    non_blank_frames = []
    """First, let's split up the output into segments in between blanks"""
    # is_blank = True
    for t in ctc_output:
        if np.argmax(t) != 0:
            non_blank_frames.append(t)
        # if is_blank and np.argmax(t) != 0:
        #     # this means we actually predict a phone here after a sequence of blanks
        #     phone_batch = []
        #     phone_batch.append(t)
        #     is_blank = False
        #     #non_blank_frames.append(t)
        # elif not is_blank and np.argmax(t) != 0:
        #     phone_batch.append(t)
        # elif not is_blank and np.argmax(t) == 0:
        #     is_blank = True
        #     non_blank_frames.append(phone_batch)
    non_blank_frames = np.asarray(non_blank_frames)
    """NEW CODE NOT IN RUN EXPERIMENTS TO IMPROVE SPEED"""
    non_blank_frames = non_blank_frames[:, 0:TOP_SYMBOLS]
    argsorts = []
    for frame in non_blank_frames:
        argsorts.append(np.flip(np.argsort(frame)))
    """Now we take first symbol to be SOS by default"""
    beam_sequences = []
    for i, frame in enumerate(non_blank_frames):
        if i == 1:
            """Create the sequences"""
            possible_prefixes = []
            for class_value, logprob in enumerate(frame):
                if class_value > 0:  # ignore the blank symbol
                    symbol = c2p[class_value]
                    if symbol != 'SOS':
                        seq = ('SOS', c2p[class_value])
                    else:
                        seq = ('SOS',)

                    seq = list(seq)
                    evaluation_seq = seq[-13:]
                    # evaluation_seq = tuple(evaluation_seq)

                    # model.score(word='b', context=('SOS', 'ax'))

                    if len(evaluation_seq) == 2:
                        context = (evaluation_seq[0],)
                        lm_score = lm2.score(word=evaluation_seq[1], context=context)
                    elif len(evaluation_seq) == 1:
                        lm_score = lm1.score(evaluation_seq[0])
                    elif len(evaluation_seq) == 3:
                        context = (evaluation_seq[0], evaluation_seq[1])
                        lm_score = lm3.score(word=evaluation_seq[2], context=context)
                    elif len(evaluation_seq) == 4:
                        context = (evaluation_seq[0], evaluation_seq[1], evaluation_seq[2])
                        lm_score = lm4.score(word=evaluation_seq[3], context=context)
                    elif len(evaluation_seq) == 5:
                        context = (evaluation_seq[0], evaluation_seq[1], evaluation_seq[2], evaluation_seq[3])
                        lm_score = lm5.score(word=evaluation_seq[4], context=context)
                    elif len(evaluation_seq) == 6:
                        context = (evaluation_seq[0], evaluation_seq[1], evaluation_seq[2],
                                   evaluation_seq[3], evaluation_seq[4])
                        lm_score = lm6.score(word=evaluation_seq[5], context=context)
                    elif len(evaluation_seq) == 7:
                        context = (evaluation_seq[0], evaluation_seq[1], evaluation_seq[2],
                                   evaluation_seq[3], evaluation_seq[4], evaluation_seq[5])
                        lm_score = lm7.score(word=evaluation_seq[6], context=context)
                    elif len(evaluation_seq) == 8:
                        context = (evaluation_seq[0], evaluation_seq[1], evaluation_seq[2],
                                   evaluation_seq[3], evaluation_seq[4], evaluation_seq[5],
                                   evaluation_seq[6])
                        lm_score = lm8.score(word=evaluation_seq[7], context=context)
                    elif len(evaluation_seq) == 9:
                        context = (evaluation_seq[0], evaluation_seq[1], evaluation_seq[2],
                                   evaluation_seq[3], evaluation_seq[4], evaluation_seq[5],
                                   evaluation_seq[6], evaluation_seq[7])
                        lm_score = lm9.score(word=evaluation_seq[8], context=context)
                    elif len(evaluation_seq) == 10:
                        context = (evaluation_seq[0], evaluation_seq[1], evaluation_seq[2],
                                   evaluation_seq[3], evaluation_seq[4], evaluation_seq[5],
                                   evaluation_seq[6], evaluation_seq[7], evaluation_seq[8])
                        lm_score = lm10.score(word=evaluation_seq[9], context=context)
                    elif len(evaluation_seq) == 11:
                        context = (evaluation_seq[0], evaluation_seq[1], evaluation_seq[2],
                                   evaluation_seq[3], evaluation_seq[4], evaluation_seq[5],
                                   evaluation_seq[6], evaluation_seq[7], evaluation_seq[8],
                                   evaluation_seq[9])
                        lm_score = lm11.score(word=evaluation_seq[10], context=context)
                    elif len(evaluation_seq) == 12:
                        context = (evaluation_seq[0], evaluation_seq[1], evaluation_seq[2],
                                   evaluation_seq[3], evaluation_seq[4], evaluation_seq[5],
                                   evaluation_seq[6], evaluation_seq[7], evaluation_seq[8],
                                   evaluation_seq[9], evaluation_seq[10])
                        lm_score = lm12.score(word=evaluation_seq[11], context=context)
                    elif len(evaluation_seq) == 13:
                        context = (evaluation_seq[0], evaluation_seq[1], evaluation_seq[2],
                                   evaluation_seq[3], evaluation_seq[4], evaluation_seq[5],
                                   evaluation_seq[6], evaluation_seq[7], evaluation_seq[8],
                                   evaluation_seq[9], evaluation_seq[10], evaluation_seq[11])
                        lm_score = lm13.score(word=evaluation_seq[12], context=context)

                    # lm_score = lm2.score(seq)
                    lm_score = np.log(lm_score + epsilon)
                    total_score = ALPHA*logprob + BETA*lm_score
                    data = {'seq': tuple(seq), 'score': total_score}
                    possible_prefixes.append(data)
            """Now we only keep top K scored sequences"""
            beam_sequences.extend(possible_prefixes)
            """https://stackoverflow.com/questions/72899/how-do-i-sort-a-list-of-dictionaries-by-a-value-of-the-dictionary"""
            sorted_beam_sequences = sorted(beam_sequences, key=lambda k: k['score'])
            sorted_beam_sequences.reverse()
            beam_sequences = sorted_beam_sequences[0:K]
        if i > 1:
            """Add on the next symbol to each of the existing beam sequences"""
            new_beam_sequences = []
            # new_beam_sequences.extend(beam_sequences)
            for seq_data in beam_sequences:
                seq = seq_data['seq']
                possible_sequences = []
                for class_value, logprob in enumerate(frame):
                    if class_value > 0:  # ignore the blank symbol
                        symbol = c2p[class_value]
                        most_recent_symbol = seq[-1]
                        if symbol != most_recent_symbol:
                            newseq = seq + (c2p[class_value],)
                        else:
                            newseq = seq
                            #total_score = seq_data['score']
                        # if len(newseq) > 2:
                        #     lm_score = lm3.score(newseq[:-3])
                        # else:
                        newseq = list(newseq)
                        evaluation_seq = newseq[-13:]
                        #evaluation_seq = tuple(evaluation_seq)

                        # model.score(word='b', context=('SOS', 'ax'))
                        if len(evaluation_seq) == 2:
                            context = (evaluation_seq[0],)
                            lm_score = lm2.score(word=evaluation_seq[1], context=context)
                        elif len(evaluation_seq) == 1:
                            lm_score = lm1.score(evaluation_seq[0])
                        elif len(evaluation_seq) == 3:
                            context = (evaluation_seq[0], evaluation_seq[1])
                            lm_score = lm3.score(word=evaluation_seq[2], context=context)
                        elif len(evaluation_seq) == 4:
                            context = (evaluation_seq[0], evaluation_seq[1], evaluation_seq[2])
                            lm_score = lm4.score(word=evaluation_seq[3], context=context)
                        elif len(evaluation_seq) == 5:
                            context = (evaluation_seq[0], evaluation_seq[1], evaluation_seq[2], evaluation_seq[3])
                            lm_score = lm5.score(word=evaluation_seq[4], context=context)
                        elif len(evaluation_seq) == 6:
                            context = (evaluation_seq[0], evaluation_seq[1], evaluation_seq[2],
                                       evaluation_seq[3], evaluation_seq[4])
                            lm_score = lm6.score(word=evaluation_seq[5], context=context)
                        elif len(evaluation_seq) == 7:
                            context = (evaluation_seq[0], evaluation_seq[1], evaluation_seq[2],
                                       evaluation_seq[3], evaluation_seq[4], evaluation_seq[5])
                            lm_score = lm7.score(word=evaluation_seq[6], context=context)
                        elif len(evaluation_seq) == 8:
                            context = (evaluation_seq[0], evaluation_seq[1], evaluation_seq[2],
                                       evaluation_seq[3], evaluation_seq[4], evaluation_seq[5],
                                       evaluation_seq[6])
                            lm_score = lm8.score(word=evaluation_seq[7], context=context)
                        elif len(evaluation_seq) == 9:
                            context = (evaluation_seq[0], evaluation_seq[1], evaluation_seq[2],
                                       evaluation_seq[3], evaluation_seq[4], evaluation_seq[5],
                                       evaluation_seq[6], evaluation_seq[7])
                            lm_score = lm9.score(word=evaluation_seq[8], context=context)
                        elif len(evaluation_seq) == 10:
                            context = (evaluation_seq[0], evaluation_seq[1], evaluation_seq[2],
                                       evaluation_seq[3], evaluation_seq[4], evaluation_seq[5],
                                       evaluation_seq[6], evaluation_seq[7], evaluation_seq[8])
                            lm_score = lm10.score(word=evaluation_seq[9], context=context)
                        elif len(evaluation_seq) == 11:
                            context = (evaluation_seq[0], evaluation_seq[1], evaluation_seq[2],
                                       evaluation_seq[3], evaluation_seq[4], evaluation_seq[5],
                                       evaluation_seq[6], evaluation_seq[7], evaluation_seq[8],
                                       evaluation_seq[9])
                            lm_score = lm11.score(word=evaluation_seq[10], context=context)
                        elif len(evaluation_seq) == 12:
                            context = (evaluation_seq[0], evaluation_seq[1], evaluation_seq[2],
                                       evaluation_seq[3], evaluation_seq[4], evaluation_seq[5],
                                       evaluation_seq[6], evaluation_seq[7], evaluation_seq[8],
                                       evaluation_seq[9], evaluation_seq[10])
                            lm_score = lm12.score(word=evaluation_seq[11], context=context)
                        elif len(evaluation_seq) == 13:
                            context = (evaluation_seq[0], evaluation_seq[1], evaluation_seq[2],
                                       evaluation_seq[3], evaluation_seq[4], evaluation_seq[5],
                                       evaluation_seq[6], evaluation_seq[7], evaluation_seq[8],
                                       evaluation_seq[9], evaluation_seq[10], evaluation_seq[11])
                            lm_score = lm13.score(word=evaluation_seq[12], context=context)
                        if lm_score > 0 and len(evaluation_seq) == 4:
                            """This didn't get hit, so I'm not providing input to language model correctly!"""
                            stop = None
                        lm_score = np.log(lm_score + epsilon)
                        total_score = ALPHA * logprob + BETA * lm_score
                        data = {'seq': tuple(newseq), 'score': total_score}
                        possible_sequences.append(data)
                new_beam_sequences.extend(possible_sequences)
            """Now we only keep top K scored sequences"""
            """https://stackoverflow.com/questions/72899/how-do-i-sort-a-list-of-dictionaries-by-a-value-of-the-dictionary"""
            sorted_beam_sequences = sorted(new_beam_sequences, key=lambda k: k['score'])
            sorted_beam_sequences.reverse()

            """remove duplicate entries"""
            unique_sorted_beam_sequences = []
            for item in sorted_beam_sequences:
                if item not in unique_sorted_beam_sequences:
                    unique_sorted_beam_sequences.append(item)

            beam_sequences = unique_sorted_beam_sequences[0:K]
            stop = None

    best_sequence = beam_sequences[0]['seq']
    return list(best_sequence)


def get_language_model():
    if not os.path.exists('3_gram_language_model.pkl') or not os.path.exists('2_gram_language_model.pkl'):
        wordlist = joblib.load('wordlist.pkl')
        dictionary = joblib.load('dict.pkl')
        phones = joblib.load('phones.pkl')
        p2c = utils.phone2class(phones)
        c2p = utils.class2phone(phones)
        """Here, the dictionary is actually the training data for the language model,
        because those are the only possible sequences of phones"""
        training_sentences_ = [value for key, value in dictionary.items()]

        """Add SOS and EOS tokens"""
        training_sentences__ = []
        for sentence in training_sentences_:
            new_sentence = ['SOS']
            new_sentence.extend(sentence)
            new_sentence.append('EOS')
            training_sentences__.append(new_sentence)

        training_sentence_one_list = []
        for sentence in training_sentences__:
            training_sentence_one_list.extend(sentence)
        vocabulary = list(set(training_sentence_one_list))

        """Training sentences need to be list of tuples"""
        training_tuples = []
        for sentence in training_sentences__:
            training_tuples.append(tuple(sentence))

        # lm = MLE(2)
        # dummy_vocab = ['a', 'b', 'c']
        # dummy_text = [[("a", "b"), ("b", "c")]]
        # lm.fit(dummy_text, vocabulary_text=dummy_vocab)
        # # lm.fit([[("a",), ("b",), ("c",)]])
        # SCORE = lm.score("a")

        # n_grams = list(ngrams(training_sentence_one_list, n=N_GRAM))

        """1-gram model"""
        all_grams = list(everygrams(training_sentence_one_list, max_len=1))
        model1 = MLE(1)
        model1.fit(text=[all_grams], vocabulary_text=vocabulary)
        joblib.dump(model1, '1_gram_language_model.pkl')
        print("Created 1-gram model...")

        """2-gram model"""
        all_grams = list(everygrams(training_sentence_one_list, max_len=2))
        model2 = MLE(2)
        model2.fit(text=[all_grams], vocabulary_text=vocabulary)
        joblib.dump(model2, '2_gram_language_model.pkl')
        print("Created 2-gram model...")

        """3-gram model"""
        all_grams = list(everygrams(training_sentence_one_list, max_len=3))
        model3 = MLE(3)
        model3.fit(text=[all_grams], vocabulary_text=vocabulary)
        joblib.dump(model3, '3_gram_language_model.pkl')
        print("Created 3-gram model...")

        """4-gram model"""
        all_grams = list(everygrams(training_sentence_one_list, max_len=4))
        model4 = MLE(4)
        model4.fit(text=[all_grams], vocabulary_text=vocabulary)
        joblib.dump(model4, '4_gram_language_model.pkl')
        print("Created 4-gram model...")

        """5-gram model"""
        all_grams = list(everygrams(training_sentence_one_list, max_len=5))
        model5 = MLE(5)
        model5.fit(text=[all_grams], vocabulary_text=vocabulary)
        joblib.dump(model5, '5_gram_language_model.pkl')
        print("Created 5-gram model...")

        """6-gram model"""
        all_grams = list(everygrams(training_sentence_one_list, max_len=6))
        model6 = MLE(6)
        model6.fit(text=[all_grams], vocabulary_text=vocabulary)
        joblib.dump(model6, '6_gram_language_model.pkl')
        print("Created 6-gram model...")

        """7-gram model"""
        all_grams = list(everygrams(training_sentence_one_list, max_len=7))
        model7 = MLE(7)
        model7.fit(text=[all_grams], vocabulary_text=vocabulary)
        joblib.dump(model7, '7_gram_language_model.pkl')
        print("Created 7-gram model...")

        """8-gram model"""
        all_grams = list(everygrams(training_sentence_one_list, max_len=8))
        model8 = MLE(8)
        model8.fit(text=[all_grams], vocabulary_text=vocabulary)
        joblib.dump(model8, '8_gram_language_model.pkl')
        print("Created 8-gram model...")

        """9-gram model"""
        all_grams = list(everygrams(training_sentence_one_list, max_len=9))
        model9 = MLE(9)
        model9.fit(text=[all_grams], vocabulary_text=vocabulary)
        joblib.dump(model9, '9_gram_language_model.pkl')
        print("Created 9-gram model...")

        """10-gram model"""
        all_grams = list(everygrams(training_sentence_one_list, max_len=10))
        model10 = MLE(10)
        model10.fit(text=[all_grams], vocabulary_text=vocabulary)
        joblib.dump(model10, '10_gram_language_model.pkl')
        print("Created 10-gram model...")

        """11-gram model"""
        all_grams = list(everygrams(training_sentence_one_list, max_len=11))
        model11 = MLE(11)
        model11.fit(text=[all_grams], vocabulary_text=vocabulary)
        joblib.dump(model11, '11_gram_language_model.pkl')
        print("Created 11-gram model...")

        """12-gram model"""
        all_grams = list(everygrams(training_sentence_one_list, max_len=12))
        model12 = MLE(12)
        model12.fit(text=[all_grams], vocabulary_text=vocabulary)
        joblib.dump(model12, '12_gram_language_model.pkl')
        print("Created 12-gram model...")

        """13-gram model"""
        all_grams = list(everygrams(training_sentence_one_list, max_len=13))
        model13 = MLE(13)
        model13.fit(text=[all_grams], vocabulary_text=vocabulary)
        joblib.dump(model13, '13_gram_language_model.pkl')
        print("Created 13-gram model...")

        """https://stackoverflow.com/questions/6462709/nltk-language-model-ngram-calculate-the-prob-of-a-word-from-context"""
        """Scroll down a little for relevant answer (you must provide n-1 context for the score function because
        the test symbol completes the n-gram"""
        # EOS_to_SOS_score = model.score(word='b', context=('SOS', 'ax'))
        # print(model.generate(20, random_seed=7))
    else:
        model1 = joblib.load('1_gram_language_model.pkl')
        model2 = joblib.load('2_gram_language_model.pkl')
        model3 = joblib.load('3_gram_language_model.pkl')
        model4 = joblib.load('4_gram_language_model.pkl')
        model5 = joblib.load('5_gram_language_model.pkl')
        model6 = joblib.load('6_gram_language_model.pkl')
        model7 = joblib.load('7_gram_language_model.pkl')
        model8 = joblib.load('8_gram_language_model.pkl')
        model9 = joblib.load('9_gram_language_model.pkl')
        model10 = joblib.load('10_gram_language_model.pkl')
        model11 = joblib.load('11_gram_language_model.pkl')
        model12 = joblib.load('12_gram_language_model.pkl')
        model13 = joblib.load('13_gram_language_model.pkl')

    return {'model_1gram': model1, 'model_2gram': model2, 'model_3gram': model3,
            'model_4gram': model4, 'model_5gram': model5, 'model_6gram': model6,
            'model_7gram': model7, 'model_8gram': model8, 'model_9gram': model9,
            'model_10gram': model10, 'model_11gram': model11, 'model_12gram': model12, 'model_13gram': model13}

def check_language_models(p2c, c2p):
    models = get_language_model()
    lm1 = models['model_1gram']
    lm2 = models['model_2gram']
    lm3 = models['model_3gram']

    """Check probabilities of lm1"""
    total_probability = 0
    for key, value in c2p.items():
        score = lm1.score(value)
        total_probability += score
    """Above checks out, now let's check conditional probabilities of lm2"""
    total_conditional_probabilities = []
    for key, value in c2p.items():
        total_probability = 0
        for key_, value_ in c2p.items():
            score = lm2.score(word=value_, context=(value,))
            total_probability += score
        total_conditional_probabilities.append(total_probability)
    stop = None



def main():
    """"""
    phones = joblib.load('phones.pkl')
    p2c = utils.phone2class(phones)
    c2p = utils.class2phone(phones)
    # _ = get_language_model()
    # check_language_models(p2c=p2c, c2p=c2p)
    data_old = joblib.load('lm_debug_data.pkl')
    data = joblib.load('/home/john/Documents/School/Fall_2020/Research/DysarthriaAugment/exps/trial_1_attention_vc_one_model_fixed_iterations_CTC_TRAINING/ctc_output_predictions/M14_B3_UW51_M8.pkl')
    data['data'] = data['ctc_outputs']
    data['transcription'] = data['true_phones']
    ctc_output = data['data']
    true_transcription = data['transcription']
    models = get_language_model()
    # lm1 = models['model_1gram']
    # lm2 = models['model_2gram']
    # lm3 = models['model_3gram']
    # lm4 = models['model_4gram']
    # lm5 = models['model_5gram']
    # lm6 = models['model_6gram']
    # lm7 = models['model_7gram']
    # lm8 = models['model_8gram']
    # lm9 = models['model_9gram']
    # lm10 = models['model_10gram']
    # lm11 = models['model_11gram']
    # lm12 = models['model_12gram']
    # lm13 = models['model_13gram']

    _ = get_best_sequence(ctc_output=ctc_output, language_models=models, c2p=c2p, p2c=p2c,
                      true_transcription=true_transcription)


if __name__ == "__main__":
    main()