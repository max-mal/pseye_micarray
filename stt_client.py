#!/usr/bin/env python

"""Client using the asyncio API."""

import asyncio
from websockets.asyncio.client import connect
import sys
import json


async def client(host, token):
    async with connect(f"ws://{host}") as websocket:
        await websocket.send(token)
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            print(data)


if __name__ == "__main__":
    asyncio.run(client(sys.argv[1], sys.argv[2]))
