"""
dummy_db.py
Setup SQLite sebagai pengganti PostgreSQL untuk development.
Nanti tinggal ganti connection string ke PostgreSQL sungguhan.
Struktur tabel diasumsikan sudah ada hasil prediksi dari Airflow pipeline.
"""

import sqlite3
import pandas as pd
import random
from datetime import datetime, timedelta

DB_PATH = "retail_dummy.db"

def create_and_populate():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # -------------------------------------------------------
    # Tabel 1: customers
    # Data demografis pelanggan
    # -------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id     TEXT PRIMARY KEY,
            age             INTEGER,
            gender          TEXT,
            city            TEXT,
            is_returning    INTEGER  -- 1 = returning, 0 = new
        )
    """)

    # -------------------------------------------------------
    # Tabel 2: transactions
    # History transaksi pelanggan
    # -------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            order_id         TEXT PRIMARY KEY,
            customer_id      TEXT,
            order_date       TEXT,
            product_category TEXT,
            total_amount     REAL,
            payment_method   TEXT,
            device_type      TEXT,
            customer_rating  INTEGER,
            delivery_days    INTEGER
        )
    """)

    # -------------------------------------------------------
    # Tabel 3: predictions
    # Hasil prediksi dari Airflow pipeline (ML model output)
    # Ini tabel utama yang di-query LLM
    # -------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            customer_id          TEXT PRIMARY KEY,
            churn_probability    REAL,    -- 0.0 - 1.0
            churn_label          TEXT,    -- 'High Risk', 'Medium Risk', 'Low Risk'
            retention_segment    TEXT,    -- 'Champions', 'Loyal', 'At Risk', 'Lost'
            recency_days         INTEGER, -- hari sejak transaksi terakhir
            frequency            INTEGER, -- total transaksi
            monetary             REAL,    -- total spend
            predicted_at         TEXT     -- timestamp prediksi terakhir
        )
    """)

    conn.commit()

    # -------------------------------------------------------
    # Populate dummy data
    # -------------------------------------------------------
    cities     = ["Istanbul", "Ankara", "Izmir", "Bursa", "Antalya"]
    genders    = ["Male", "Female", "Other"]
    categories = ["Electronics", "Fashion", "Home & Garden",
                  "Sports", "Books", "Beauty", "Toys", "Food"]
    payments   = ["Credit Card", "Debit Card", "Cash", "PayPal", "Bank Transfer"]
    devices    = ["Mobile", "Desktop", "Tablet"]
    segments   = ["Champions", "Loyal", "At Risk", "Lost"]

    customers  = []
    transactions = []
    predictions  = []

    random.seed(42)
    base_date = datetime(2024, 3, 26)

    for i in range(1, 201):  # 200 dummy customers
        cid = f"CUST_{i:05d}"
        age = random.randint(18, 75)
        gender = random.choice(genders)
        city = random.choice(cities)
        is_returning = random.randint(0, 1)
        customers.append((cid, age, gender, city, is_returning))

        # 1-10 transaksi per customer
        freq = random.randint(1, 10)
        total_spend = 0
        last_order_date = None

        for j in range(freq):
            oid = f"ORD_{i:05d}_{j:02d}"
            days_ago = random.randint(1, 450)
            order_date = (base_date - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            amount = round(random.uniform(50, 5000), 2)
            total_spend += amount
            cat = random.choice(categories)
            pay = random.choice(payments)
            dev = random.choice(devices)
            rating = random.randint(1, 5)
            delivery = random.randint(1, 30)
            transactions.append((oid, cid, order_date, cat, amount,
                                  pay, dev, rating, delivery))

            # track most recent order
            if last_order_date is None or order_date > last_order_date:
                last_order_date = order_date

        # hitung recency
        last_dt = datetime.strptime(last_order_date, "%Y-%m-%d")
        recency = (base_date - last_dt).days

        # churn probability simulasi sederhana
        churn_prob = round(min(1.0, recency / 400 + random.uniform(-0.1, 0.1)), 2)
        churn_prob = max(0.0, churn_prob)

        if churn_prob >= 0.7:
            churn_label = "High Risk"
        elif churn_prob >= 0.4:
            churn_label = "Medium Risk"
        else:
            churn_label = "Low Risk"

        # retention segment berdasarkan RFM sederhana
        if recency < 30 and freq >= 5:
            segment = "Champions"
        elif recency < 90 and freq >= 3:
            segment = "Loyal"
        elif recency < 180:
            segment = "At Risk"
        else:
            segment = "Lost"

        predictions.append((
            cid, churn_prob, churn_label, segment,
            recency, freq, round(total_spend, 2),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

    # Insert semua data
    cursor.executemany(
        "INSERT OR IGNORE INTO customers VALUES (?,?,?,?,?)", customers)
    cursor.executemany(
        "INSERT OR IGNORE INTO transactions VALUES (?,?,?,?,?,?,?,?,?)", transactions)
    cursor.executemany(
        "INSERT OR IGNORE INTO predictions VALUES (?,?,?,?,?,?,?,?)", predictions)

    conn.commit()
    conn.close()
    print(f"Dummy DB created: {DB_PATH}")
    print(f"  customers   : {len(customers)} rows")
    print(f"  transactions: {len(transactions)} rows")
    print(f"  predictions : {len(predictions)} rows")

if __name__ == "__main__":
    create_and_populate()
