"""
agent.py
AI Analyst Agent — Text-to-SQL + Visualization
Menggunakan SQLite untuk dev, tinggal ganti ke PostgreSQL di production.
"""

import os
import sqlite3
import json
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

# -------------------------------------------------------
# CONFIG
# Ganti ke PostgreSQL nanti:
# DB_URI = "postgresql://user:password@localhost:5432/retail_db"
# -------------------------------------------------------
DB_PATH = "retail_dummy.db"

SCHEMA_INFO = """
Database memiliki 3 tabel berikut:

1. customers (customer_id, age, gender, city, is_returning)
   - is_returning: 1 = pelanggan lama, 0 = pelanggan baru

2. transactions (order_id, customer_id, order_date, product_category,
                 total_amount, payment_method, device_type,
                 customer_rating, delivery_days)

3. predictions (customer_id, churn_probability, churn_label,
                retention_segment, recency_days, frequency,
                monetary, predicted_at)
   - churn_label  : 'High Risk', 'Medium Risk', 'Low Risk'
   - retention_segment: 'Champions', 'Loyal', 'At Risk', 'Lost'
   - churn_probability: nilai 0.0 - 1.0
   - Tabel ini adalah hasil prediksi ML dari pipeline Airflow

Untuk analisis churn, retensi dan untuk pertanyaan tentang 'Risk' atau 'Segment', WAJIB pakai tabel `predictions`.
Untuk data demografis, Jika butuh 'City' atau 'Gender' saat analisa churn, lakukan JOIN: 
`FROM predictions p JOIN customers c ON p.customer_id = c.customer_id` dengan tabel customers.
Untuk data transaksi, JOIN dengan tabel transactions.
"""


# -------------------------------------------------------
# HELPER: jalankan SQL ke database
# -------------------------------------------------------
def run_sql(query: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    # Untuk PostgreSQL nanti:
    # from sqlalchemy import create_engine
    # engine = create_engine(DB_URI)
    # df = pd.read_sql(query, engine)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


# -------------------------------------------------------
# TOOL 1: SQL Query Tool
# -------------------------------------------------------
@tool
def query_data(sql: str) -> str:
    """Gunakan tool ini untuk mengambil data dari database
    dengan SQL query. Input harus berupa SQL query yang valid.
    Selalu gunakan tool ini ketika user membutuhkan data,
    angka, statistik, atau informasi dari database."""
    try:
        df = run_sql(sql)
        if df.empty:
            return "Hasil query kosong. Coba cek filter WHERE kamu, mungkin terlalu ketat."
        # Kembalikan dalam format JSON string agar bisa dibaca tool create_chart
        return df.to_json(orient="records")
    except Exception as e:
        # Jika error, kirim pesan error SQL-nya ke LLM agar dia bisa benerin query-nya
        return f"SQL Error: {str(e)}. Tolong periksa nama kolom atau syntax JOIN kamu."


# -------------------------------------------------------
# TOOL 2: Visualization Tool
# LLM yang generate SQL-nya sendiri via query_data,
# tool ini hanya bertanggung jawab render chart dari
# data yang sudah ada.
# -------------------------------------------------------
@tool
def create_chart(payload: dict) -> str:
    """
    Gunakan tool ini untuk membuat visualisasi.
    payload harus berisi:
    {
        "data_json": "...",
        "chart_type": "bar/pie/line/scatter",
        "title": "...",
        "x_col": "...",
        "y_col": "..."
    }
    """
    try:
        import json
        import streamlit as st
        import pandas as pd

        data_json = payload["data_json"]
        chart_type = payload["chart_type"]
        title = payload["title"]
        x_col = payload["x_col"]
        y_col = payload["y_col"]

        data = json.loads(data_json)
        df = pd.DataFrame(data)

        if df.empty:
            return "Tidak ada data untuk divisualisasikan."

        chart_data = {
            "type": chart_type,
            "title": title,
            "columns": [x_col, y_col],
            "data": df.to_dict(orient="records")
        }

        st.session_state.pending_chart = chart_data
        
        return f"CHART_READY:{title}"

    except Exception as e:
        return f"Error saat menyiapkan chart: {str(e)}"

# -------------------------------------------------------
# SETUP AGENT
# -------------------------------------------------------
SYSTEM_PROMPT = f"""Kamu adalah AI Retail Analyst untuk perusahaan e-commerce.
Tugasmu membantu stakeholder (CMO, CEO) memahami data pelanggan, terutama terkait churn dan retensi pelanggan.

==============================
PERILAKU WAJIB
==============================

1. Kamu TIDAK bisa membuat gambar atau grafik sendiri.
2. Grafik hanya bisa dibuat dengan memanggil tool `create_chart`.
3. Jika user meminta grafik atau visualisasi, kamu WAJIB memanggil tool.
4. Jika tidak memanggil tool, sistem akan gagal.

==============================
URUTAN KERJA WAJIB
==============================

Jika user meminta grafik:

STEP 1 → Panggil query_data
STEP 2 → Gunakan output JSON dari query_data
STEP 3 → Panggil create_chart dengan payload yang valid
STEP 4 → Setelah tool dipanggil, baru berikan interpretasi bisnis

JANGAN berhenti sebelum memanggil create_chart.

==============================
FORMAT TOOL CALL WAJIB
==============================

Action: create_chart
Action Input:
{{
  "data_json": "...",
  "chart_type": "bar/pie/line/scatter",
  "title": "...",
  "x_col": "...",
  "y_col": "..."
}}

Jika format ini tidak diikuti, sistem akan gagal.

==============================
ATURAN SQL
==============================

- Gunakan GROUP BY untuk grafik
- Jangan ambil ribuan baris mentah
- Untuk churn/segment gunakan tabel predictions
- Gunakan JOIN jika butuh city/gender

==============================
INFORMASI DATABASE
==============================

{SCHEMA_INFO}
Gunakan Bahasa Indonesia profesional.
"""
load_dotenv()

def create_agent():
    llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )
    tools = [query_data, create_chart]
    agent = create_react_agent(llm, tools)
    return agent

def invoke_agent(agent, user_input: str) -> str:
    result = agent.invoke(
        {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),  # ← pisahkan system prompt
            HumanMessage(content=user_input)        # ← hanya pertanyaan user
            ]
        },
        config={"recursion_limit":50}
    )
    return result["messages"][-1].content
