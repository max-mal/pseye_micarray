import asyncio
import json
import queue
import threading
import sounddevice as sd
import numpy as np
from argparse import ArgumentParser
from websockets.asyncio.server import ServerConnection, serve
from vosk import Model, KaldiRecognizer

from beamformer import Beamformer


class SttServer:
    MODEL_PATH = "vosk-model-small-ru-0.22"
    SAMPLE_RATE = 16000
    CHANNELS = 4
    BLOCK_DURATION = 0.5

    def __init__(self, model_path=None, device=None, token='changeme'):
        self.bf = Beamformer(fs=self.SAMPLE_RATE)

        self.device = device if device is not None else "default"
        self.token = token

        model_path = model_path if model_path is not None else self.MODEL_PATH
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, self.SAMPLE_RATE)
        self.recognizer.SetWords(True)

        self.audio_queue = queue.Queue()
        self.loop_ready = threading.Event()
        self.ws_clients = []

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)

        theta, audio = self.bf.process(indata)
        # print(f"theta={theta}")

        self.audio_queue.put((theta, audio))

    def stt_loop(self):
        while True:
            theta, data = self.audio_queue.get()
            pcm16 = (data * 32767).astype(np.int16).tobytes()

            if self.recognizer.AcceptWaveform(pcm16):
                result = json.loads(self.recognizer.Result())
                if result.get("text"):
                    self.broadcast({
                        "type": "final",
                        "text": result["text"],
                        "theta": np.rad2deg(theta),
                    })
            else:
                partial = json.loads(self.recognizer.PartialResult())
                if partial.get("partial"):
                    self.broadcast({
                        "type": "partial",
                        "text": partial["partial"],
                        "theta": np.rad2deg(theta),
                    })

    async def ws_handler(self, websocket: ServerConnection):
        print("Client added!")
        try:
            token = await websocket.recv()
            print(f"<<< Client connected")

            if token == self.token:
                print(f"<<< Client authenticated")
                self.ws_clients.append(websocket)
                await websocket.wait_closed()
                self.ws_clients.remove(websocket)

            await websocket.close()
        except Exception as e:
            print(e)

    async def _broadcast(self, message):
        print(message)
        print("sending to clients")

        data = json.dumps(message)
        for client in self.ws_clients:
            client: ServerConnection
            await client.send(data)

    def broadcast(self, message):
        asyncio.run_coroutine_threadsafe(
            self._broadcast(message),
            self.loop
        )

    async def sockets_server(self):
        self.loop = asyncio.get_running_loop()
        self.loop_ready.set()

        host = "0.0.0.0"
        print(f"🌐 WebSocket server on ws://{host}:2700")
        async with serve(self.ws_handler, host, 2700) as server:
            self.server = server
            await server.serve_forever()

    def start_sockets_server(self):
        asyncio.run(self.sockets_server())

    def main(self):
        print("🎤 Starting microphone...")
        stream = sd.InputStream(
            device=self.device,
            samplerate=self.SAMPLE_RATE,
            blocksize=int(self.BLOCK_DURATION * self.SAMPLE_RATE),
            channels=self.CHANNELS,
            callback=self.audio_callback
        )
        stream.start()

        th = threading.Thread(
            target=self.start_sockets_server,
            daemon=True,
        )
        th.start()

        self.loop_ready.wait()
        self.stt_loop()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-m', '--model')
    parser.add_argument('-d', '--device')
    parser.add_argument('-t', '--token')

    args = parser.parse_args()

    server = SttServer(
        model_path=args.model,
        device=args.device,
        token=args.token,
    )
    server.main()
