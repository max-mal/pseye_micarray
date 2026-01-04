import numpy as np
import sounddevice as sd
import time
import argparse
import soundfile as sf
import os

from plotter import DoaPlotter
from beamformer import Beamformer


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--device', required=False)
parser.add_argument('-f', '--file', required=False)
parser.add_argument('-t', '--theta', required=False)

args = parser.parse_args()

if args.device is None and args.file is None:
    print("Device or file is needed")
    print(sd.query_devices())


channels = 4
fs = 16000
block_duration = 0.5

bf = Beamformer(fft_size=512)

if args.device:
    plotter = DoaPlotter()
    plotter.start()

processed_data = np.array([])


def callback(audio, _frames, _time, status):
    global processed_data

    if status:
        print(status)

    if not len(audio):
        pass

    tstart = time.monotonic()

    theta, result = bf.process(audio, steer=args.theta)
    if theta is not None:
        theta_deg = np.rad2deg(theta)
        print(f"theta={theta_deg}, result.shape={result.shape}")

        if args.device:
            plotter.put(theta_deg)

    if args.file:
        processed_data = np.concatenate((processed_data, result))

    tend = time.monotonic()
    print("time: ", tend - tstart)


if args.device is not None:
    with sd.InputStream(
        device=args.device,
        channels=channels,
        callback=callback,
        blocksize=int(fs * block_duration),
        samplerate=fs
    ):
        while True:
            if input() == 'q':
                break

if args.file is not None:
    data = np.load(args.file)
    part_length = int(fs * block_duration)
    parts = [data[i:i+part_length] for i in range(0, len(data), part_length)]

    for part in parts:
        callback(part, None, None, None)

    print("RMS:", np.sqrt(np.mean(processed_data**2)))
    sf.write(
        f"processed-{os.path.basename(args.file)}.wav", processed_data, fs)
