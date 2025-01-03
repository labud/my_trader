import json
import datetime

def formatDepth(msg):
    depth = json.loads(msg)
    time = datetime.datetime.fromtimestamp(depth['E']/1000).strftime('%Y-%m-%d %H:%M:%S')
    print(time)
    buys = depth['b']
    print("buy:")
    for buy in buys:
        if float(buy[1]) * float(buy[0]) > 10000:
            print(buy)
    print("sell")
    sells = depth['a']
    for sell in sells:
        if float(sell[1]) * float(sell[0]) > 10000:
            print(sell)
    print()
