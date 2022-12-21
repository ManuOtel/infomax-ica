import wave, gc

import numpy as np
import matplotlib.pyplot as plt

from ica import ica1
from typing import Tuple
from pathlib import Path

def wave_read(filename: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(filename), 'rb') as f:
        buffer = f.readframes(f.getnframes())
        inter = np.frombuffer(buffer, dtype=f'int{f.getsampwidth()*8}')
        return np.reshape(inter, (-1, f.getnchannels())), f.getframerate()

def get_time_wave(audio, sr, length, pad):
    time = np.expand_dims(np.arange(0,length,1/sr), axis=1)
    if pad:
        p_audio = np.zeros((length*sr, 1))
        p_audio[:np.shape(audio)[0]] = audio
        audio = p_audio 
    t_audio = np.transpose(np.hstack((audio,time)))
    return t_audio

if __name__ == '__main__':
    # Play all files in the current directory
    path = 'wav_example.wav'
    audio, sr = wave_read(path)
    t_audio = get_time_wave(audio, sr, 5, True)
    plt.subplot(3,1,1)
    plt.plot(t_audio[1,:], t_audio[0,:])

    A,S,W = ica1(t_audio, 2)
    plt.subplot(3,1,2)
    plt.plot(S[0,:])
    plt.subplot(3,1,3)
    plt.plot(S[1,:])
    plt.show()
