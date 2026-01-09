import numpy as np
import sounddevice as sd
import time
import argparse
import soundfile as sf
import os
import queue

from plotter import DoaPlotter
from beamformer import Beamformer


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--front', required=True)
parser.add_argument('-r', '--rear', required=True)

args = parser.parse_args()


channels = 4
fs = 16000
block_duration = 0.5

# 2 PS Eye, looking in opposite direction, 5cm spacing between
mic_positions = np.array([
    [-0.03, -0.01,  0.01,  0.03, 0.03,  0.01, -0.01, -0.03],
    [0,  0,  0,  0, 0.05,  0.05,  0.05,  0.05]
])

bf = Beamformer(
    mic_positions=mic_positions,
    mic_count=8,
    fft_size=512,
    channels_map=[1, 3, 2, 0, 5, 7, 6, 4],
    normalize_forward=False,
)


qf = queue.Queue()
qr = queue.Queue()

plotter = DoaPlotter()
plotter.start()


def process_audio():
    front_audio = qf.get()
    rear_audio = qr.get()

    audio = np.hstack((front_audio, rear_audio))
    print(audio.shape)

    theta, b = bf.process(audio)
    if theta is not None:
        theta_deg = np.rad2deg(theta)
        plotter.put(theta_deg)


def callback_front(audio, _frames, _time, status):
    if status:
        print(status)

    qf.put(audio)


def callback_rear(audio, _frames, _time, status):
    if status:
        print(status)

    qr.put(audio)


stream_front = sd.InputStream(
    device=args.front,
    channels=channels,
    callback=callback_front,
    blocksize=int(fs * block_duration),
    samplerate=fs
)

stream_rear = sd.InputStream(
    device=args.rear,
    channels=channels,
    callback=callback_rear,
    blocksize=int(fs * block_duration),
    samplerate=fs
)

stream_front.start()
stream_rear.start()

while True:
    process_audio()
