import librosa
import numpy as np
import os
import joblib
import multiprocessing
import concurrent.futures
from tqdm import tqdm
from preprocessing import collect_files
import yaml
from easydict import EasyDict as edict
import pysptk
from pysptk.synthesis import Synthesizer, MLSADF
import matplotlib.pyplot as plt

config = edict(yaml.load(open('config.yml'), Loader=yaml.SafeLoader))

class Mel_log_spect(object):
    def __init__(self):
        self.nfft = config.data.fftl
        self.num_mels = config.data.num_mels
        self.hop_length = config.data.hop_length
        self.top_db = config.data.top_db
        self.sr = config.data.sr

    def feature_normalize(self, x):
        log_min = np.min(x)
        x = x - log_min
        x = x / self.top_db
        x = x.T
        return x

    def get_Mel_log_spect(self, y):
        y = librosa.util.normalize(S=y)
        spect = librosa.feature.melspectrogram(y=y, sr=self.sr, n_fft=self.nfft,
                                               hop_length=self.hop_length, n_mels=self.num_mels)
        log_spect = librosa.core.amplitude_to_db(spect, ref=1.0, top_db=self.top_db)
        log_spect = self.feature_normalize(log_spect)
        return log_spect

    def norm_Mel_log_spect_to_amplitude(self, feature):
        feature = feature * self.top_db
        spect = librosa.core.db_to_amplitude(feature, ref=1.0)
        return spect

    def audio_from_spect(self, feature):
        spect = self.norm_Mel_log_spect_to_amplitude(feature)
        audio = librosa.feature.inverse.mel_to_audio(spect.T, sr=self.sr, n_fft=self.nfft, hop_length=self.hop_length)
        return audio

    def convert_and_write(self, load_path, write_path):
        y, sr = librosa.core.load(path=load_path, sr=self.sr)
        feature = self.get_Mel_log_spect(y, n_mels=self.num_mels)
        audio = self.audio_from_spect(feature)
        librosa.output.write_wav(write_path, y=audio, sr=self.sr, norm=True)

class MCEP(object):
    def __init__(self, need_synth):
        self.frame_length = config.data.fftl
        self.hop_length = config.data.hop_length
        self.sr = config.data.sr
        self.order = config.data.mcep_order
        self.alpha = config.data.mcep_alpha
        if need_synth:
            self.build_synth()
        else:
            self.synthesizer = None

    def build_synth(self):
        self.synthesizer = Synthesizer(MLSADF(order=self.order, alpha=self.alpha), self.hop_length)

    def get_MCEP(self, utterance):
        utterance = librosa.util.normalize(utterance)
        utterance = utterance + np.random.normal(loc=0, scale=0.0000001, size=utterance.shape[0])
        utterance = librosa.util.normalize(utterance)
        utterance = utterance.astype(np.float64)  # necessary for synthesizer
        frames = librosa.util.frame(utterance, frame_length=self.frame_length, hop_length=self.hop_length).astype(np.float64).T
        # Windowing
        frames *= pysptk.blackman(self.frame_length)
        assert frames.shape[1] == self.frame_length
        # Pitch
        pitch = pysptk.swipe(utterance.astype(np.float64), fs=self.sr, hopsize=self.hop_length, min=60, max=240, otype="pitch")
        mcep = pysptk.mcep(frames, self.order, self.alpha)
        return mcep, pitch

    def synthesize_from_MCEP(self, mcep, pitch):
        mcep = mcep.copy(order='C')  # fixes "ndarray not C-contiguous error
        b = pysptk.mc2b(mcep, self.alpha)
        excitation = pysptk.excite(pitch.astype(np.float64), self.hop_length)
        x = self.synthesizer.synthesis(excitation.astype(np.float64), b.astype(np.float64))
        return x

def process(file):
    try:
        audio, _ = librosa.core.load(file, sr=config.data.sr)
        feature_processor = Mel_log_spect()
        """We need MCEP features to find dtw path for Mel log spect features"""
        mcep_extractor = MCEP(need_synth=False)
        mcep, pitch = mcep_extractor.get_MCEP(audio)
        features = feature_processor.get_Mel_log_spect(audio)
        features = features[3:-4, :]  # 7 more frames from features than mcep
        mel_log_dump_path = os.path.join(config.directories.features, file.split('/')[-1][:-4] + '.pkl')
        mcep_dump_path = os.path.join(config.directories.mcep, file.split('/')[-1][:-4] + '.pkl')

        # plt.subplot(211)
        # plt.imshow(mcep[:, 1:].T)
        # plt.subplot(212)
        # plt.imshow(features.T)
        # plt.show()

        joblib.dump(features, mel_log_dump_path)
        joblib.dump({'mcep': mcep, 'pitch': pitch}, mcep_dump_path)

    except:
        print("Had trouble processing file " + file + " ...")

def main():
    if not os.path.isdir(config.directories.features):
        os.mkdir(config.directories.features)
    if not os.path.isdir(config.directories.mcep):
        os.mkdir(config.directories.mcep)
    files = collect_files(config.directories.silence_removed)
    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for _ in tqdm(executor.map(process, files)):
            """"""

if __name__ == "__main__":
    main()