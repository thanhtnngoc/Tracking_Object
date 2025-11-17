import asyncio
import websockets
import time
import threading
import json

def UIThread(connection):
    connection.run()

class UIConnection:

    _sendQueue = []
    _lock = asyncio.Lock()
    _active = False

    _onDataCallback = None

    def __init__(self):
        thread = threading.Thread(target=UIThread, args=(self,))
        thread.start()

        self._active = True

    async def msgHandler(self, websocket):
        sendTask = asyncio.create_task(self.sendHandler(websocket))
        receiveTask = asyncio.create_task(self.receiveHandler(websocket))
        await asyncio.wait([sendTask, receiveTask], return_when=asyncio.FIRST_COMPLETED)


    async def sendHandler(self, websocket):
        while True:
            await asyncio.sleep(0.05)

            if len(self._sendQueue) == 0:
                continue

            message = '\n'.join(self._sendQueue)

            try:
                await websocket.send(message)
            except websockets.exceptions.ConnectionClosedOK:
                print('connection closed during send')

            self._sendQueue = []

    async def receiveHandler(self, websocket):
        async for message in websocket:
            await self.handleMessage(message)
            if message:
                # Gửi phản hồi lại cho client
                reply = {"status": "ok", "echo": message}
                await websocket.send(json.dumps(reply))
                print("📤 Sent reply:", reply)
            else:
                print("⚠️ Empty message received")

    async def handleMessage(self, data: str):
        if self._onDataCallback == None:
            return

        # newline split json. Convert to object then pass up-chain
        messages = data.split('\n')

        for message in messages:
            try:
                self._onDataCallback(json.loads(message))
            except Exception as e:
                print(e)

    async def main(self):
        self.stop_event = threading.Event()
        stop = asyncio.get_event_loop().run_in_executor(None, self.stop_event.wait)

        async with websockets.serve(self.msgHandler, '127.0.0.1', 5678):
            await stop


    def run(self):
        asyncio.run(self.main())

    def stop(self):
        self.stop_event.set()
        self._active = False

    def active(self):
        return self._active

    def send(self, data):
        self._sendQueue.append(json.dumps(data))

    def setOnDataCallback(self, callback):
        self._onDataCallback = callback

if __name__ == '__main__':
    connection = UIConnection()

    while True:
        try:
            time.sleep(1)
        except:
            break

    connection.stop()