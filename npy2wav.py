import numpy as np
import sys
import os
import soundfile as sf


filename = sys.argv[1]
framerate = int(sys.argv[2])
data = np.load(filename)

# data_interleaved = data.T.flatten()

sf.write(f"{filename}.wav", data, framerate)
