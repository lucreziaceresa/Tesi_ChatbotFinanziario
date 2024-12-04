import mysql.connector
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import tqdm

def normalize_output(x):
    min_val = x.min()
    max_val = x.max()
    normalized_x = (x - min_val) / (max_val - min_val)
    return normalized_x

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="chatbot"
)

cursor = db.cursor()
cursor.execute('SELECT * FROM scraping_news')
scraping_news = cursor.fetchall()

cursor = db.cursor()
cursor.execute('SELECT * FROM finbert')
finbert = cursor.fetchall()

if len(finbert) != 0:
    finbert_text = [item[1] for item in finbert]
    table_to_insert = []
    for item in scraping_news:
        if item[2] not in finbert_text:
            table_to_insert.append(item)
else:
    table_to_insert = scraping_news



if len(table_to_insert) != 0:

    tokenizer = AutoTokenizer.from_pretrained("finbert_local/tokenizer/")
    model = AutoModelForSequenceClassification.from_pretrained("finbert_local/model/")

    texts = [item[2] for item in table_to_insert]
    date_times = [item[1] for item in table_to_insert]

    positive = []
    negative = []
    neutral = []
    positive_norm = []
    negative_norm = []
    neutral_norm = []

    for text in texts:
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        output = model(**inputs).logits
        output = output.detach().numpy()[0]

        positive.append(output[0])
        negative.append(output[1])
        neutral.append(output[2])

        output_norm = normalize_output(output)

        positive_norm.append(output_norm[0])
        negative_norm.append(output_norm[1])
        neutral_norm.append(output_norm[2])


    for d, t, pos, neg, neu, pos_n, neg_n, neu_n in zip(date_times, texts, positive, negative, neutral, positive_norm, negative_norm, neutral_norm):
        outputs = (d, t, float(pos), float(neg), float(neu), float(pos_n), float(neg_n), float(neu_n))
        cursor.execute('INSERT INTO finbert (date_time, text, positive, negative, neutral, positive_norm, negative_norm, neutral_norm) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)', tuple(outputs))
        db.commit()

cursor.close()
db.close()