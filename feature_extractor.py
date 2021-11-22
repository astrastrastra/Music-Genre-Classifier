import numpy as np
from scipy.io import wavfile
from scipy.signal import get_window
import scipy.fftpack as fft
from matplotlib import pyplot as plt

def setup():
    frame_size = 800
    overlap = 0.2
    mel_filter_count = 10
    dct2mfcc_count = 39

    return frame_size, overlap, mel_filter_count, dct2mfcc_count

def framing(samples, frame_size, overlap):
    samples = np.pad(samples, int(np.floor(frame_size / 2.0)), mode='reflect')
    hopSize = int(frame_size - np.floor(overlap * frame_size))
    frames = np.array([(samples[i:i+frame_size]) for i in range(0, len(samples)-frame_size, hopSize)])

    return frames


def mel_filter_point_coords(fmin, fmax, mel_filter_num, frame_size, sample_rate):
    fmin_mel = 2595.0 * np.log10(1.0 + fmin / 700.0)
    fmax_mel = 2595.0 * np.log10(1.0 + fmax / 700.0)

    def mel_to_freq(mels):
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num + 2)
    freqs = mel_to_freq(mels)

    filter_points = np.floor((frame_size + 1) / sample_rate * freqs).astype(int)

    return filter_points, freqs


def mel_filters_construct(filter_points, frame_size):
    filters = np.zeros((len(filter_points) - 2, int(frame_size / 2 + 1)))

    for n in range(len(filter_points) - 2):
        filters[n, filter_points[n]: filter_points[n + 1]] = np.linspace(0, 1, filter_points[n + 1] - filter_points[n])
        filters[n, filter_points[n + 1]: filter_points[n + 2]] = np.linspace(1, 0, filter_points[n + 2] - filter_points[
            n + 1])

    return filters


def dct(dct_filter_num, filter_len):
    dct_iii = np.empty((dct_filter_num, filter_len))
    dct_iii[0, :] = 1.0 / np.sqrt(filter_len)

    samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)

    for i in range(1, dct_filter_num):
        dct_iii[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)

    return dct_iii


def extractMFCC(title):

    frame_size, overlap,  mel_filter_count, dct_filter_count = setup()

    sample_rate, audio = wavfile.read(title)

    audio_framed = framing(audio, frame_size, overlap)
    window = get_window("hann", frame_size, fftbins=True)
    audio_win = audio_framed * window
    audio_winT = np.transpose(audio_win)

    audio_fft = np.empty((int(1 + frame_size // 2), audio_winT.shape[1]), dtype=np.complex64, order='F')

    for n in range(audio_fft.shape[1]):
        audio_fft[:, n] = fft.fft(audio_winT[:, n])[:audio_fft.shape[0]]

    audio_fft = np.transpose(audio_fft)
    audio_log = 10.0 * np.log10(audio_fft)
    audio_power = np.square(np.abs(audio_log))

    freq_min = 0
    freq_high = sample_rate / 2


    filter_points, mel_freqs = mel_filter_point_coords(freq_min, freq_high, mel_filter_count, frame_size, sample_rate)
    filters = mel_filters_construct(filter_points, frame_size)

    enorm = 2.0 / (mel_freqs[2:mel_filter_count + 2] - mel_freqs[:mel_filter_count])
    filters *= enorm[:, np.newaxis]

    audio_filtered = np.dot(filters, np.transpose(audio_power))

    dct_filters = dct(dct_filter_count, mel_filter_count)

    mfcc = np.dot(dct_filters, audio_filtered)

    fea=[]
    fea.extend(np.mean(mfcc, axis=1))# Avg
    fea.extend(np.var(mfcc, axis=1)) # StD
    fea.extend(np.min(mfcc, axis=1)) # Min
    fea.extend(np.max(mfcc, axis=1)) # Max

    return fea
