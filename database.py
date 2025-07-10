import sqlite3
import json
from datetime import datetime

def init_db():
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS requests (
        id INTEGER PRIMARY KEY,
        timestamp TEXT NOT NULL,
        filename TEXT,
        result_img TEXT,
        giraffe_count INTEGER)''')
    conn.commit()
    conn.close()

def log_request(filename, result_img, count):
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute('''INSERT INTO requests (timestamp, filename, result_img, giraffe_count)
                 VALUES (?, ?, ?, ?)''', 
              (datetime.now(), filename, result_img, count))
    conn.commit()
    conn.close()