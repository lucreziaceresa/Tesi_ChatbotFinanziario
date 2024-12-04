import finnhub
from datetime import datetime, timezone, timedelta
import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('FINNHUB_API')
finnhub_client = finnhub.Client(api_key=api_key)

finnhub_oneyear = finnhub_client.company_news('AAPL', _from=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'), to=datetime.now().strftime('%Y-%m-%d'))

dates = []
sources = []
texts = []
for i,item in enumerate(finnhub_oneyear):
    source = item['source']
    sources.append(source)
    text = item['summary']
    texts.append(text)

    date = item['datetime']
    if date == -62135596800:
        dates.append(dates[-1])
    else:        
        date = datetime.fromtimestamp(date, tz=timezone.utc)
        dates.append(date.strftime('%Y-%m-%d %H:%M'))

db_table = [(s,d,t) for s,d,t in zip(sources, dates, texts)]

db = mysql.connector.Connect(
    host = 'localhost',
    user = 'root',
    password = '',
    database = 'chatbot' 
)

cursor = db.cursor()

cursor.execute('SELECT * FROM finnhub_aapl')
finnhub_aapl = cursor.fetchall()

if len(finnhub_aapl) != 0:
    stored_text = [item[2] for item in finnhub_aapl]
    filtered_table = []
    for item in db_table:
        if item[2] not in stored_text:
            filtered_table.append(item)

    db_table = filtered_table

for item in db_table:
    cursor.execute('INSERT INTO finnhub_aapl (source, datetime, text) VALUES (%s,%s,%s)', item)

db.commit()

cursor.close()
db.close()