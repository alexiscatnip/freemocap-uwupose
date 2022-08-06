import asyncio
import threading
from collections import deque

import socketio

import freemocap
from freemocap import inference_gui, session
from src import steamVR_thread

sio = socketio.AsyncClient()

"""
client, subscribes to a topic published by the server.
"""

PORT = int("8000")
network_output_thread = None
pipe_3d_points_in = deque(maxlen=1)


@sio.event
async def connect():
    print('connection established')

@sio.event
async def my_message(data):
    print('message received with ', data)
    pump_data_to_SteamVR_Thread(data)

@sio.event
async def disconnect():
    print('disconnected from server')


def pump_data_to_SteamVR_Thread(data):
    # change(data)


    pipe_3d_points_in.append(data)


async def main():
    sesh = session.Session()

    pipe_HMD_out = deque(maxlen=1)
    network_output_thread = steamVR_thread.SteamVRThread(
        pipe_3d_points_in,
        pipe_HMD_out,
        sesh
    )

    # start UI thread
    gui_thread = threading.Thread(target=inference_gui.make_inference_gui,
                                  args=(session,),
                                  daemon=True)
    gui_thread.start()

    print ("now waiting for packets from port 5000")
    await sio.connect('http://localhost:5000')
    await sio.wait()

asyncio.run(main())
