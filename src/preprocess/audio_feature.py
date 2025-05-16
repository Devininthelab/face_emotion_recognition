import os, warnings
import cv2
import shutil
import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
from decord import VideoReader
from moviepy.editor import VideoFileClip

from scipy.io import wavfile # scipy library to read wav files
import numpy as np
from scipy.fftpack import dct
from matplotlib import pyplot as plt
from PIL import Image

INPUT_SIZE = 224
NUM_FRAME = 8
SAMPLING_RATE = 6

# Notes: No normalization is done for the video frames, values are in the range [0, 255] as the torch version is different from tf


def normalize_audio(audio):
    """
    Normalize the audio signal to the range [-1, 1].
    Args:
        audio (numpy.ndarray): The audio signal.
    Returns:
        numpy.ndarray: The normalized audio signal.
    """
    audio = audio / np.max(np.abs(audio))
    return audio

def MFCC(signal, sample_rate):
    """
    Compute the MFCC features from the audio signal.
    Args:
        signal (numpy.ndarray): The audio signal.
        sample_rate (int): The sample rate of the audio signal.
    Returns:
        numpy.ndarray: The MFCC features.
    """
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    frame_size = 0.025
    frame_stride = 0.0001

    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    frames *= np.hamming(frame_length)
    NFFT = 512

    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
    nfilt = 40

    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB
    num_ceps = 13
    mfcc = dct(filter_banks, type = 2, axis=1, norm="ortho")[:,1: (num_ceps + 1)] # keep 2-13
    cep_lifter = 22
    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n/ cep_lifter)
    mfcc *= lift
    return mfcc

def format_frames(frame, output_size):
    """
    Convert a frame (as NumPy array) to uint8 and resize it using PIL.
    Returns the resized frame as a NumPy array.
    
    Args:
        frame: np.ndarray of shape (H, W, C), dtype float32 or uint8
        output_size: tuple (height, width)
    
    Returns:
        np.ndarray of shape (output_height, output_width, C), dtype uint8
    """
    # Convert to uint8 if necessary
    if frame.dtype != np.uint8:
        frame = (frame * 255).astype(np.uint8)

    # Convert to PIL image
    image = Image.fromarray(frame)

    # Resize
    image_resized = image.resize(output_size[::-1], Image.BILINEAR)  # PIL uses (width, height)

    # Convert back to numpy
    return np.array(image_resized)

def read_video(file_path):
    vr = VideoReader(file_path)
    frames = vr.get_batch(range(len(vr))).asnumpy()
    return format_frames(
        frames,
        output_size=(INPUT_SIZE, INPUT_SIZE)
    )


