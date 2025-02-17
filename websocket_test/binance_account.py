import base64
import time
import json
from cryptography.hazmat.primitives.serialization import load_pem_private_key
import websocket

API_KEY = "hy2gyV4g5XeHYQ3hYJBFZ3DfvxwamL9npS3K7rHdbpYZDStP90Apwur1wyJhI6ex"
PRIVATE_KEY_PATH = "/Users/liuhua2/.ssh/pem/binance/Ed25519_Private_key.txt"

def getSinature(params):
    with open(PRIVATE_KEY_PATH, 'rb') as f:
        private_key = load_pem_private_key(data=f.read(), password=None)
    # 参数中加时间戳：
    timestamp = int(time.time() * 1000)  # 以毫秒为单位的 UNIX 时间戳
    params['timestamp'] = timestamp
    # 参数中加签名：
    payload = '&'.join([f'{param}={value}' for param, value in sorted(params.items())])
    signature = base64.b64encode(private_key.sign(payload.encode('ASCII')))
    params['signature'] = signature.decode('ASCII')
    return params

def getAcccountInfo(ws):
    # get account infomation
    params = {
        "apiKey": API_KEY,
        "omitZeroBalances": "true"
    }
    params = getSinature(params)
    request = {
        "id": "1",
        "method": "account.status",
        "params": params
    }

    ws.send(json.dumps(request))
    result = ws.recv()
    return result

def main():
    ws = websocket.create_connection('wss://ws-api.binance.com:443/ws-api/v3')

    times = []
    for i in range(0, 10):
        start = time.time()
        result = getAcccountInfo(ws)
        end = time.time()
        print(end -  start)
        times.append(end - start)
    max_value = max(times)
    min_value = min(times)
    avg_value = 0 if len(times) == 0 else sum(times) / len(times)
    print("max_value:", max_value)
    print("min_value:", min_value)
    print("avg_value", avg_value)


if __name__ == "__main__":
    main()