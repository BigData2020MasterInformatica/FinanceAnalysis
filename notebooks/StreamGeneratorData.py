#!/usr/bin/env python
# coding: utf-8

import websocket
import json
import datetime
import time;

fileName = "streamCSV"
apiToken = ""

def on_message(ws, message):
    global fileName
    
    # save message data
    print(message)
    jsonD = json.loads(message)
    f = open(fileName, "a")
    for dato in jsonD['data']:
        f.write(str(dato['s']) + ',' + str(dato['v']) + ',' + str(datetime.datetime.fromtimestamp(dato['t']/1000.0)) + '\n')
    f.close()
    

def on_error(ws, error):
    print(error)


def on_close(ws):
    print("### closed ###")


def on_open(ws):
    print('### open ###')
    global fileName
    # subscribe to enterprises data
    ws.send('{"type":"subscribe","symbol":"AAPL"}')
    ws.send('{"type":"subscribe","symbol":"AMZN"}')
    ws.send('{"type":"subscribe","symbol":"BINANCE:BTCUSDT"}')
    ws.send('{"type":"subscribe","symbol":"IC MARKETS:1"}')
    
    # create file with timestamp
    ts = time.time()
    fileName = "streamCSV" + str(ts) + ".txt"
    f = open(fileName, "a")
    f.write('name,value,time\n')
    f.close()


if __name__ == "__main__":

    # connect to API socket
    ws = websocket.WebSocketApp("wss://ws.finnhub.io?token=" + str(apiToken), 
                              on_message = on_message,
                              on_error = on_error,
                              on_close = on_close)
    ws.on_open = on_open
    ws.run_forever()
