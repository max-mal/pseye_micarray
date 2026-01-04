import numpy as np
import pyroomacoustics as pra
import sounddevice as sd
import sys
import time
import soundfile as sf
import os


fs = 16000
channels = 4
duration = 5
fft_size = 512
hop = fft_size // 2

block_duration = 0.5  # sec

# PS Eye geometry (meters)
mic_spacing = 0.02
mic_positions = pra.linear_2D_array([0, 0], 4, 0, mic_spacing)
print(mic_positions)

# mic_positions = np.array([
#     [0.0, 0.02, 0.04, 0.06],  # x
#     [0.0, 0.0,  0.0,  0.0],   # y
# ])

if len(sys.argv) < 2:
    print("usage: main.py <device number>")
    print("available devices")
    print(sd.query_devices())
    sys.exit(1)

doa = pra.doa.NormMUSIC(
    mic_positions,
    fs,
    fft_size,
)

bf = pra.Beamformer(
    mic_positions,
    fs=fs,
    N=fft_size,
    # Lg=np.ceil(0.1*fs)
)

processed_data = np.array([])
prev_block = np.empty((channels, 0))


def callback(audio, frames, _time, status):
    global processed_data, prev_block

    if not len(audio):
        pass

    tstart = time.monotonic()

    # indata shape: (frames, channels)
    X = audio.T  # -> (channels, samples)
    # reorder to physical mic order
    X = X[[1, 3, 2, 0], :]

    if prev_block.shape[1] > 0:
        data = np.hstack((prev_block, X))
    else:
        data = X

    bf.record(data, fs=fs)

    data_stft = np.vstack((prev_block.T, X.T))
    X_stft = pra.transform.stft.analysis(data_stft, fft_size, hop).T

    prev_block = X.copy()

    doa.locate_sources(X_stft)
    if doa.azimuth_recon is not None:
        print("  Recovered azimuth:", doa.azimuth_recon / np.pi * 180.0, "degrees")
        theta = doa.azimuth_recon[0]
        bf.far_field_weights(theta)

    bf.far_field_weights(np.deg2rad(90))
    output = bf.process()[prev_block.shape[1]:]
    print(output.shape)

    processed_data = np.concatenate((processed_data, output))
    print(processed_data.shape)

    tend = time.monotonic()
    print("time: ", tend - tstart)


print("Запись...")
filename = sys.argv[1]

data = np.load(filename)
part_length = int(fs * block_duration)
parts = [data[i:i+part_length] for i in range(0, len(data), part_length)]

for part in parts:
    callback(part, None, None, None)

# callback(data, None, None, None)
sf.write(f"processed-{os.path.basename(filename)}.wav", processed_data, fs)
