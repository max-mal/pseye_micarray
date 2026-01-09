import numpy as np
import pyroomacoustics as pra
import sys


class Beamformer:
    def __init__(
        self,
        mic_positions=None,
        fs=16000,
        fft_size=1024,
        mic_count=4,  # PS Eye linear array
        mic_spacing=0.02,
        channels_map=[1, 3, 2, 0],
        normalize_forward=True,
    ):
        self.fs = fs
        self.fft_size = fft_size
        self.fft_hop = fft_size // 2
        self.current_theta = None
        self.normalize_forward = normalize_forward

        if mic_positions is None:
            self.mic_positions = pra.linear_2D_array(
                [0, 0], mic_count, 0, mic_spacing)
        else:
            self.mic_positions = mic_positions

        self.channels_map = channels_map

        print("Configured array positions:")
        print(self.mic_positions)

        self.prev_block = np.empty((mic_count, 0))

        self.doa = pra.doa.NormMUSIC(
            self.mic_positions,
            self.fs,
            self.fft_size,
        )

        self.bf = pra.Beamformer(
            self.mic_positions,
            fs=self.fs,
            N=self.fft_size,
        )
        self.steer(np.deg2rad(0))

    def steer(self, theta):
        if self.current_theta == theta:
            return

        self.bf.far_field_weights(theta)

    def normalize_theta(self, theta):
        theta_deg = np.rad2deg(theta)
        if theta_deg > 180:
            return theta

        return np.deg2rad(90 - theta_deg + 270)

    def process(self, audio, steer=None):
        X = audio.T  # -> (channels, samples)

        # reorder to physical mic order
        if self.channels_map is not None:
            X = X[self.channels_map, :]

        # append previous block for filters warmup
        if self.prev_block.shape[1] > 0:
            data = np.hstack((self.prev_block, X))
        else:
            data = X

        self.bf.record(data, fs=self.fs)  # feed frames to beamformer

        theta = None
        # data_stft = np.vstack((self.prev_block.T, X.T))
        data_stft = X.T
        X_stft = pra.transform.stft.analysis(
            data_stft, self.fft_size, self.fft_hop)

        self.prev_block = X.copy()

        if X_stft is not None:
            X_stft = X_stft.T

            self.doa.locate_sources(X_stft)
            if self.doa.azimuth_recon is not None:
                theta = self.doa.azimuth_recon[0]
                if self.normalize_forward:
                    theta = self.normalize_theta(theta)

                if steer is None:
                    self.steer(theta)

        if steer is not None:
            self.steer(np.deg2rad(float(steer)))

        output = self.bf.process()
        return (theta, output[self.prev_block.shape[1]:])


if __name__ == '__main__':
    import sounddevice as sd

    fs = 16000
    block_duration = 0.1

    bf = Beamformer(fs=fs)

    def callback(audio, _, __, status):
        if status:
            print(status)

        if not len(audio):
            pass

        theta, result = bf.process(audio)
        if theta is not None:
            print(f"theta={np.rad2deg(theta)}, result.shape={result.shape}")

    with sd.InputStream(
        device=sys.argv[1],
        channels=4,
        callback=callback,
        blocksize=int(fs * block_duration),
        samplerate=fs
    ):
        print("Press 'q' and enter to exit")
        while True:
            if input() == 'q':
                break
