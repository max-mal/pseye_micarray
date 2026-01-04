import sounddevice as sd
import numpy as np
import sys
import datetime

samplerate = 48000
channels = 4
duration = 5

if len(sys.argv) < 2:
    print("usage: rec.py <device number>")
    print("available devices")
    print(sd.query_devices())
    sys.exit(1)

name = sys.argv[2] if len(sys.argv) >= 3 else 'rec'

print("Запись...")
audio = sd.rec(
    int(duration * samplerate),
    samplerate=samplerate,
    channels=channels,
    dtype='float32',
    device=sys.argv[1]  # device number, sd.query_devices()
)
sd.wait()

date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%I:%S")
print(audio.shape)  # (samples, 4)
np.save(f"{name}-{date}.npy", audio)
