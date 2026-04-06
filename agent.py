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

Untuk analisis churn dan retensi, selalu gunakan tabel predictions.
Untuk data demografis, JOIN dengan tabel customers.
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
            return "Query berhasil tapi tidak ada data yang ditemukan."
        # Kembalikan max 20 baris supaya tidak membanjiri context LLM
        result = df.head(20).to_string(index=False)
        return f"Hasil query ({len(df)} baris total):\n{result}"
    except Exception as e:
        return f"Error saat menjalankan query: {str(e)}"


# -------------------------------------------------------
# TOOL 2: Visualization Tool
# LLM yang generate SQL-nya sendiri via query_data,
# tool ini hanya bertanggung jawab render chart dari
# data yang sudah ada.
# -------------------------------------------------------
@tool
def create_chart(sql: str, chart_type: str, title: str, x_col: str, y_col: str) -> str:
    """Gunakan tool ini HANYA ketika user meminta grafik,
    chart, atau visualisasi. Tool ini akan menjalankan SQL
    dan membuat chart dari hasilnya.

    Parameter:
    - sql       : SQL query untuk ambil data chart (kamu yang generate)
    - chart_type: jenis chart, pilih salah satu: 'bar', 'pie', 'line', 'scatter'
    - title     : judul chart yang deskriptif
    - x_col     : nama kolom untuk sumbu X (atau label pie)
    - y_col     : nama kolom untuk sumbu Y (atau nilai pie)

    Contoh penggunaan:
    sql="SELECT churn_label, COUNT(*) as jumlah FROM predictions GROUP BY churn_label",
    chart_type="bar", title="Distribusi Churn Risk", x_col="churn_label", y_col="jumlah"
    """
    try:
        df = run_sql(sql)
        if df.empty:
            return "Tidak ada data untuk divisualisasikan."

        # Validasi kolom yang diminta LLM ada di hasil query
        if x_col not in df.columns or y_col not in df.columns:
            available = ", ".join(df.columns.tolist())
            return f"Kolom tidak ditemukan. Kolom tersedia: {available}"

        chart_data = {
            "type"   : chart_type,
            "title"  : title,
            "x_col"  : x_col,
            "y_col"  : y_col,
            "data"   : df[[x_col, y_col]].to_dict(orient="records"),
            "columns": [x_col, y_col]
        }

        # Simpan ke Streamlit session_state — lebih reliable dari file JSON
        try:
            import streamlit as st
            st.session_state.pending_chart = chart_data
        except Exception:
            # Fallback jika tidak ada Streamlit context (misal test di terminal)
            import json, tempfile, os
            path = os.path.join(tempfile.gettempdir(), "chart_data.json")
            with open(path, "w") as f:
                json.dump(chart_data, f)

        return f"CHART_READY:{title}"

    except Exception as e:
        return f"Error membuat chart: {str(e)}"


# -------------------------------------------------------
# SETUP AGENT
# -------------------------------------------------------
SYSTEM_PROMPT = f"""Kamu adalah AI Retail Analyst untuk perusahaan e-commerce.
Tugasmu membantu stakeholder (CMO, CEO) memahami data pelanggan,
terutama terkait churn dan retensi pelanggan.

{SCHEMA_INFO}

Panduan menjawab:
- Selalu gunakan tool query_data untuk mengambil data
- Jika user minta grafik/chart, gunakan tool create_chart
  dengan SQL yang kamu generate sendiri sesuai kebutuhan
- Untuk create_chart, tentukan chart_type yang paling sesuai:
  * 'bar'     → perbandingan antar kategori
  * 'pie'     → proporsi/persentase
  * 'line'    → trend waktu
  * 'scatter' → korelasi dua variabel
- Jawab dalam Bahasa Indonesia yang mudah dipahami non-teknis
- Tambahkan interpretasi bisnis dari data yang ditemukan
- Jangan tampilkan SQL query mentah ke user
- Jika data tidak ditemukan, jelaskan dengan sopan
"""
load_dotenv()

def create_agent():
    llm = ChatGroq(
        model="llama3-8b-8192",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )
    tools = [query_data, create_chart]
    agent = create_react_agent(llm, tools)
    return agent

def invoke_agent(agent, user_input: str) -> str:
    full_input = f"""{SYSTEM_PROMPT}

    Pertanyaan user: {user_input}"""

    result = agent.invoke({
        "messages": [HumanMessage(content=full_input)]  # ← full_input bukan user_input
    })
    return result["messages"][-1].content
