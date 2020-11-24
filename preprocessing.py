import os
from tqdm import tqdm
import joblib
import librosa
import noisereduce as nr
import multiprocessing
import concurrent.futures
import soundfile as sf
import silence_removal
import yaml
from easydict import EasyDict as edict

config = edict(yaml.load(open('config.yml'), Loader=yaml.SafeLoader))

STAGE = '01'
"""Stage 0 is removing stationary noise from recordings"""
"""Stage 1 is removing silence from recordings"""

def collect_files(directory):
    all_files = []
    for path, subdirs, files in tqdm(os.walk(directory)):
        for name in files:
            filename = os.path.join(path, name)
            all_files.append(filename)
    return all_files

def collect_noise(audio_sample):
    """Use the first half second of audio as noise sample for stationary noise removal"""
    noise = audio_sample[0:8000]
    return noise

def filter(path):
    try:
        data, rate = librosa.core.load(path, sr=config.data.sr)
        noise = collect_noise(data)
        reduced_noise = nr.reduce_noise(audio_clip=data, noise_clip=noise, verbose=False)
        reduced_noise = librosa.util.normalize(reduced_noise)  # normalize from -1 to 1
        dump_path = config.directories.noise_removed + path.split('/')[-1]
        sf.write(dump_path, reduced_noise, config.data.sr, subtype='PCM_16')
    except:
        print("Trouble processing file " + path + ' ...')

def keep_wav(files):
    new_files = []
    for f in files:
        if '.wav' in f:
            new_files.append(f)
    return new_files

def main(stage='01'):
    if '0' in stage:
        if not os.path.isdir(config.directories.noise_removed):
            os.mkdir(config.directories.noise_removed)
        """Collect files to preprocess by speaker"""
        for speaker in config.data.speakers:
            folder = os.path.join(config.directories.root, speaker)
            files = collect_files(folder)
            joblib.dump(files, os.path.join(config.directories.speakers, speaker + '.pkl'))

        """Combine into one list of files to process using multiprocessing"""
        for i, speaker in enumerate(collect_files(config.directories.speakers)):
            if i == 0:
                files = joblib.load(speaker)
            else:
                files.extend(joblib.load(speaker))
        files = keep_wav(files)  # make sure all files are .wav files

        """Remove noise and save audio files"""
        with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            for _ in tqdm(executor.map(filter, files)):
                """"""
    if '1' in stage:
        if not os.path.isdir(config.directories.silence_removed):
            os.mkdir(config.directories.silence_removed)
        """Remove silence from recordings"""
        silence_removal.main()

if __name__ == "__main__":
    if not os.path.isdir(config.directories.speakers):
        os.mkdir(config.directories.speakers)
    main(stage=STAGE)