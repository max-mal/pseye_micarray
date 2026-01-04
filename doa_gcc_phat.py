import sounddevice as sd
import numpy as np
import sys

from plotter import DoaPlotter

MIC_DIST = 0.06  # m
C = 343          # sound speed, m/s


def gcc_phat(sig, refsig, fs):
    n = sig.shape[0] + refsig.shape[0]
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)
    cc = np.fft.irfft(R / (np.abs(R) + 1e-10), n=n)
    max_shift = n // 2
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift]))
    shift = np.argmax(cc) - max_shift
    return shift / fs


def estimate_angle(audio, fs=16000):
    tau = gcc_phat(audio[:, 0], audio[:, 1], fs)

    max_tau = MIC_DIST / C
    tau = np.clip(tau, -max_tau, max_tau)

    angle = np.arcsin(tau * C / MIC_DIST)
    return np.degrees(angle)


if __name__ == '__main__':
    samplerate = 16000
    channels = 4
    duration = 0.5

    if len(sys.argv) < 2:
        print("usage: doa_gcc_phat.py <device number>")
        print("available devices")
        print(sd.query_devices())
        sys.exit(1)

    plotter = DoaPlotter()
    plotter.start()

    print("Запись...")
    while True:
        audio = sd.rec(
            int(duration * samplerate),
            samplerate=samplerate,
            channels=channels,
            dtype='float32',
            device=sys.argv[1]  # device number, sd.query_devices()
        )
        sd.wait()

        print(audio.shape)  # (samples, 4)
        angle = estimate_angle(audio, samplerate)
        print(angle)
        plotter.put(angle)

        for i in range(4):
            print(f"Mic {i} RMS:", np.sqrt(np.mean(audio[:, i]**2)))
