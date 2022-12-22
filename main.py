import wave, gc

import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

from sklearn.decomposition import FastICA
from scipy.io import wavfile
from ica import ica1, pca_whiten
from typing import Tuple
from pathlib import Path

def wave_read(filename: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(filename), 'rb') as f:
        buffer = f.readframes(f.getnframes())
        inter = np.frombuffer(buffer, dtype=f'int{f.getsampwidth()*8}')
        return np.reshape(inter, (-1, f.getnchannels())), f.getframerate()

def get_time_wave(audio: np.ndarray, sr, length, pad):
    time = np.expand_dims(np.arange(0,length,1/sr), axis=1)
    if pad:
        p_audio = np.zeros((length*sr, 1), dtype=np.float64)
        p_audio[:np.shape(audio)[0]] = audio
    pp_audio = np.pad(audio.squeeze(), (0,(sr*length-np.shape(audio)[0])), 'constant', constant_values=(0))
    pp_audio2 = np.expand_dims(pp_audio, axis=1)
    t_audio = np.hstack((pp_audio2, time)).T
    #print(pp_audio.shape)
    return t_audio, p_audio.T, pp_audio2.T

if __name__ == '__main__':
    # Play all files in the current directory
    path = 'wav_example.wav'
    audio, sr = wave_read(path)

    t_audio, p_audio, pp_audio = get_time_wave(audio, sr, 5, True)

    print(sr)
    print(audio.shape)
    print(t_audio.shape)
    print(p_audio.shape)
    print(pp_audio.shape)

    #sd.play(pp_audio[0, :], sr)

    #res = t_audio[0,:len(audio)]-audio
    # print(t_audio[0,:len(audio)].shape)
    # print(audio.squeeze().shape)
    res = t_audio[0,:len(audio)]-audio.squeeze()
    for i in res:
        if i != 0:
            print(i)
    #A,S,W = ica1(pp_audio, 2)

    ica = FastICA(n_components=2, whiten='unit-variance', max_iter=int(1e4))
    S = ica.fit_transform(t_audio.T)

    print(S.shape)
    print(S)

    plt.subplot(4,1,1)
    plt.plot(t_audio[1,:len(audio)], audio)
    plt.subplot(4,1,2)
    plt.plot(t_audio[1,:], p_audio[0,:])
    plt.subplot(4,1,3)
    plt.plot(t_audio[1,:], pp_audio[0,:])
    plt.subplot(4,1,4)
    plt.plot(S[:, 0], S[:, 1])
    plt.show()

    # print(S[0,:5000].shape)

    sd.play(S[1,:], sr)

    plt.plot(S[:, 0], S[:, 1])
    plt.show()
    
    sd.play(S[0,:], sr)

    wavfile.write('reconstructed_1.wav', sr, S[:5000,0])
    wavfile.write('reconstructed_2.wav', sr, S[:5000,1])

    plt.plot(S[:, 0], S[:, 1])
    plt.show()
