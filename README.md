### PS Eye mic array experiments

#### Direction of origin estimation
- GCC-PHAT
	`python3 doa_gcc_phat.py <device>`

- MUSIC (pyroomacoustics)
	`python3 main.py -d <device>`

#### Beamforming (pyroomacoustics)
1. record audio
	`python3 rec.py <device>`
2. process recorded (performs beamforming by DOA angle)
	`python3 main.py --file <recorded file>`
	Also, can use --theta option to beamform by specific angle (deg.)

#### Speech recognition (Vosk)
- Download vosk model (vosk-model-small-ru-0.22 by default)
- Run `python3 stt_server.py <device>`

#### PS Eye microphones physical layout

Channels: 4
Inter-mic spacing: 2cm.

Channel-to-mic mapping:
```
|------------------------|
|    X    X    X    X    |
|    1    3    2    0    |
|              <-2cm->   |
|------------------------|
      |             |
      |    FRONT    |
      |             |
      |-------------|
```
