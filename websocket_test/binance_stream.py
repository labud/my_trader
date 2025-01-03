import websocket
import json

from websocket_test.utils.formatter import formatDepth


def on_open(ws):
    print("opened")
    data = {
      "method": "SUBSCRIBE",
      "params": [
        "btcusdt@depth"
      ],
      "id": 1
    }
    ws.send(json.dumps(data))


def on_message(ws, message):
    formatDepth(message)

def on_close(ws, code, reason):
    print("closed due to ", reason, " with response code ", code)

def on_error(ws, error):
    print("error: ", error)


def main():
    ws = websocket.WebSocketApp(url = "wss://stream.binance.com:9443/ws",
                                     on_open = on_open,
                                     on_message = on_message,
                                     on_close = on_close,
                                     on_error = on_error)
    ws.run_forever(ping_interval=15)

if __name__ == "__main__":
    main()