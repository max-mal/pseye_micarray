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

bf_front = Beamformer(fft_size=512)
bf_rear = Beamformer(fft_size=512)

qf = queue.Queue()
qr = queue.Queue()

plotter = DoaPlotter()
plotter.start()


def process_audio():
    front_audio = qf.get()
    rear_audio = qr.get()

    tf, bf = bf_front.process(front_audio)
    tr, br = bf_rear.process(rear_audio)

    tf_deg = np.rad2deg(tf)
    tr_deg = np.rad2deg(tr)

    rms_f = np.sqrt(np.mean(bf**2))
    rms_r = np.sqrt(np.mean(br**2))

    print(tf_deg, tr_deg)
    print(rms_f, rms_r)

    if rms_f > rms_r:
        theta = tr_deg
    else:
        theta = tr_deg - 180

    plotter.put(theta)

    front_audio = None
    rear_audio = None


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
