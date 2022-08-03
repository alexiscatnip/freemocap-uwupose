import uvicorn
from fastapi import FastAPI
from fastapi import APIRouter, WebSocket

from fastapi_websocket_pubsub import PubSubEndpoint

app = FastAPI()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
define server endpoints.
"""
@app.get("/")
async def root():
    return{"msg": "Hello, the server is up."}

@app.websocket("/pose")
async def pose(
    web_socket: WebSocket
):
    await web_socket.accept()
    print("client has connected.")

    while True:
        data = await web_socket.receive_json()
        print(data)

