import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient

app = FastAPI()

"""
define client's API.
"""

def test_websocket():
    client = TestClient(app)
    with client.websocket_connect("/pose") as websocket:
        data = {"Hello, the server is up."}
        websocket.send_json(data);

def test_read_main():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Hello, the server is up."}

if __name__ == "__main__":
    test_read_main()
    test_websocket()
