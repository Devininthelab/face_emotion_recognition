import os, warnings
import cv2
import shutil
import numpy as np
import pandas as pd
import torch
import librosa

import matplotlib.pyplot as plt
from decord import VideoReader
from moviepy.editor import VideoFileClip

from scipy.io import wavfile # scipy library to read wav files
import numpy as np
from scipy.fftpack import dct
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from io import BytesIO

INPUT_SIZE = 224
NUM_FRAME = 8
SAMPLING_RATE = 6

# Notes: No normalization is done for the video frames, values are in the range [0, 255] as the torch version is different from tf

# There is probably typo in the orginal code, as the frame stride is 0.0001, which is not possible
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
    Dense overlapp: # 25 ms, 0.1 ms stride
    Use hanning window to reduce the edge effects.
    Number of FFT: 512
    Number of MEL filters: 40
    Number of MFCC coefficients: 13
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


############ Utils ######################################3
def save_frame_as_uint8_image(frame, filename):
    # Convert RGB frame (float) to uint8 and save
    frame_uint8 = np.clip(frame * 255, 0, 255).astype(np.uint8) if frame.dtype == np.float32 or frame.max() <= 1 else frame.astype(np.uint8)
    frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, frame_bgr)

def save_mfcc_as_rgb_uint8_image(audio, sr, output_path, cmap='viridis'):
    audio = normalize_audio(audio)
    mfcc = MFCC(audio, sr)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
    cax = ax.matshow(
        np.transpose(mfcc),
        interpolation="nearest",
        aspect="auto",
        cmap=plt.cm.afmhot_r,
        origin="lower",
    )
    plt.axis('off')
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # Close the figure to avoid displaying it
    buf.seek(0)

    img = Image.open(buf).convert('RGB')
    img.save(output_path)

    # # Step 2: Normalize MFCC to 0-1 for colormap
    # mfcc_min = mfcc.min()
    # mfcc_max = mfcc.max()
    # mfcc_norm = (mfcc - mfcc_min) / (mfcc_max - mfcc_min + 1e-6)  # Prevent div by 0

    # # Step 3: Map to RGB using a colormap (matplotlib)
    # colormap = cm.get_cmap(cmap)
    # mfcc_rgb = colormap(mfcc_norm)[:, :, :3]  # Drop alpha channel if present

    # # Step 4: Convert to uint8 (0-255)
    # mfcc_rgb_uint8 = (mfcc_rgb * 255).astype(np.uint8)

    # # Step 5: Convert to PIL image and save
    # img = Image.fromarray(mfcc_rgb_uint8)
    # img.save(output_path)

def extract_frames_and_mfccs(video_path, output_dir, num_segments=8):
    os.makedirs(output_dir, exist_ok=True)

    clip = VideoFileClip(video_path)
    duration = clip.duration
    audio, sr = librosa.load(video_path, sr=48000)

    segment_duration = duration / num_segments

    for i in range(num_segments):
        time = (i + 0.5) * segment_duration
        frame = clip.get_frame(time)
        frame_filename = os.path.join(output_dir, f"frame_{i}.jpg")
        save_frame_as_uint8_image(frame, frame_filename)

        # Extract MFCC for the segment
        start_sample = int(i * segment_duration * sr)
        end_sample = int((i + 1) * segment_duration * sr)
        segment_audio = audio[start_sample:end_sample]
        mfcc_filename = os.path.join(output_dir, f"mfcc_{i}.jpg")
        save_mfcc_as_rgb_uint8_image(segment_audio, sr, mfcc_filename, cmap='viridis')
    clip.close()
    return output_dir

