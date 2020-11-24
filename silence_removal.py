import collections
import contextlib
import sys
import wave
import webrtcvad
import librosa
import soundfile
import os
import numpy as np
from tqdm import tqdm
import multiprocessing
import concurrent.futures
import preprocessing
import soundfile as sf
import yaml
from easydict import EasyDict as edict

config = edict(yaml.load(open('config.yml'), Loader=yaml.SafeLoader))

temp_folder = './temp_folder_concatenated_audio/'
if not os.path.isdir(temp_folder):
    os.mkdir(temp_folder)


def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

def resample_audio(path):
    wav = librosa.load(path)
    wav = librosa.core.resample(wav[0], orig_sr=wav[1], target_sr=config.data.sr)
    soundfile.write(file=path, data=wav, samplerate=config.data.sr)

def read_wave(path):
    """Reads a .wav file.

    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate

def write_wave(path, audio, sample_rate):
    """Writes a .wav file.

    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)

class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.

    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.

    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.

    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.

    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.

    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.

    Arguments:

    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).

    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])

def process_file(file):
    try:
        blockPrint()  # there's some weird byte printing stuff here with segment, very annoying
        sample = file.split('/')[-1][:-4]
        args = [3, file]
        resample_audio(args[1])
        audio, sample_rate = read_wave(args[1])

        vad = webrtcvad.Vad(int(args[0]))
        frames = frame_generator(30, audio, sample_rate)
        frames = list(frames)
        segments = vad_collector(sample_rate, 30, 300, vad, frames)

        for i, segment in enumerate(segments):
            sample = file.split('/')[-1][:-4]
            if not os.path.isdir(os.path.join(temp_folder, sample)):
                os.mkdir(os.path.join(temp_folder, sample))  # Necessary for multiprocessing, need separate folders
            path = os.path.join(temp_folder, sample, sample + '_chunk-%002d.wav' % (i,))
            write_wave(path, segment, sample_rate)

        """Now we must concatenate the chunks and write them to local_folder"""
        temp_location = os.path.join(temp_folder, sample)
        for path, subdirs, files in os.walk(temp_location):
            for i, name in enumerate(files):
                filepath = os.path.join(path, name)
                audio, _ = librosa.core.load(path=filepath, sr=16000)
                if i == 0:
                    full_audio = audio
                else:
                    full_audio = np.concatenate((full_audio, audio))
        final_write_path = os.path.join(config.directories.silence_removed, sample + '.wav')
        full_audio = librosa.util.normalize(full_audio)
        sf.write(final_write_path, full_audio, 16000, "PCM_16")
        enablePrint()
    except:
        """"""

def main():
    """Directory creation"""
    if not os.path.isdir(config.directories.silence_removed):
        os.mkdir(config.directories.silence_removed)


    """Process the files using all cpus"""
    all_files = preprocessing.collect_files(config.directories.noise_removed)
    process_file(all_files[0])
    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for _ in tqdm(executor.map(process_file, all_files)):
            """"""

if __name__ == '__main__':
    main()
