import websocket
import json
from datetime import datetime, timezone
import mysql.connector
import os
from dotenv import load_dotenv
import finnhub
import time

load_dotenv()
api_key = os.getenv('FINNHUB_API')
finnhub_client = finnhub.Client(api_key=api_key)

while True:
    try:
        market_status = finnhub_client.market_status(exchange='US')
        if market_status:
            break
    except:
        time.sleep(20)

if market_status['isOpen'] == True:
    symbol_stock = '"AAPL"'

    msg = None

    def on_message(ws, message):
        global msg
        msg = message
        ws.close()

    def on_error(ws, error):
        print(error)

    def on_close(ws):
        print("### closed ###")

    def on_open(ws):
        send_string = '{"type":"subscribe","symbol":' + f'{symbol}' + '}'
        ws.send(send_string)

    if __name__ == "__main__":
        websocket.enableTrace(False)
        symbol = symbol_stock
        ws = websocket.WebSocketApp(f"wss://ws.finnhub.io?token={api_key}",
                                on_message = on_message,
                                on_error = on_error,
                                on_close = on_close)
        ws.on_open = on_open
        ws.run_forever()

    json_object = json.loads(msg)

    data = json_object['data']

    if type(data) == list and len(data) != 0:
        most_recent_trade = max(data, key=lambda x: x["t"])
        most_recent_timestamp = most_recent_trade['t'] // 1000
        most_recent_time = datetime.fromtimestamp(most_recent_timestamp, tz=timezone.utc)
        most_recent_price = most_recent_trade["p"]


        db = mysql.connector.Connect(
            host = 'localhost',
            user = 'root',
            password = '',
            database = 'chatbot' 
        )

        cursor = db.cursor()

        cursor.execute('INSERT INTO finnhub_price (symbol, datetime, price) VALUES (%s, %s,%s)', (symbol[1:-1], most_recent_time.strftime('%Y-%m-%d %H:%M'), most_recent_price))

        db.commit()

        cursor.close()
        db.close()
else:
    print('The market is closed')
