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
def create_chart(data_json: str, chart_type: str, title: str, x_col: str, y_col: str) -> str:
    """Gunakan tool ini untuk visualisasi setelah data didapat."""
    try:
        # Parse JSON hasil dari query_data
        df = pd.read_json(data_json)

        if df.empty:
            return "Tidak ada data untuk divisualisasikan."

        clean_data = df.to_dict(orient="records")

        chart_data = {
            "type": chart_type,
            "title": title,
            "columns": [x_col, y_col],
            "data": clean_data
        }

        import streamlit as st
        st.session_state.pending_chart = chart_data
        
        return f"CHART_READY:{title}"
        
    except Exception as e:
        return f"Error saat menyiapkan chart: {str(e)}"

# -------------------------------------------------------
# SETUP AGENT
# -------------------------------------------------------
SYSTEM_PROMPT = f"""Kamu adalah AI Retail Analyst untuk perusahaan e-commerce.
Tugasmu membantu stakeholder (CMO, CEO) memahami data pelanggan,
terutama terkait churn dan retensi pelanggan.

Jika user meminta grafik/visualisasi, kamu wajib menjalankan dua langkah: (1) Ambil data dengan query_data, 
lalu (2) Gunakan hasil JSON dari data tersebut sebagai input untuk create_chart. Jangan berhenti sebelum memanggil create_chart.

Jangan pernah mengambil data mentah ribuan baris untuk grafik. 
Gunakan GROUP BY di SQL agar data yang dikirim ke tool visualisasi sudah ringkas.

{SCHEMA_INFO}

URUTAN KERJA (WAJIB):
1. Jika user bertanya tentang data/statistik, panggil tool `query_data`.
2. Jika user meminta "grafik", "visualisasi", "tampilkan", atau "diagram":
   - LANGKAH A: Panggil `query_data` untuk mendapatkan data mentah (format JSON).
   - LANGKAH B: Ambil OUTPUT JSON dari langkah A, lalu panggil `create_chart` menggunakan data tersebut.
   - LANGKAH C: Berikan interpretasi bisnis singkat berdasarkan grafik yang muncul.

ATURAN KETAT:
- JANGAN pernah memberikan data angka saja jika user meminta grafik.
- Tool `create_chart` MEMBUTUHKAN data JSON dari `query_data`. Jangan mengarang data.
- Gunakan Bahasa Indonesia yang profesional namun mudah dimengerti.
- Jangan tampilkan query SQL mentah kepada user kecuali diminta untuk debugging.
- Jika data dari `query_data` kosong, beritahu user dan jangan panggil `create_chart`.

CONTOH ALUR:
User: "Tampilkan grafik komposisi segmen retensi."
Assistant: 
1. Call `query_data(sql="SELECT retention_segment, COUNT(*) as jumlah FROM predictions GROUP BY retention_segment")`
2. (Terima JSON data)
3. Call `create_chart(data_json='...', chart_type='pie', title='Komposisi Segmen Retensi', x_col='retention_segment', y_col='jumlah')`
4. Response: "Berikut adalah grafik komposisi segmen pelanggan Anda..."

ATURAN VISUALISASI:
- Bar Chart: Gunakan untuk perbandingan kategori (contoh: Churn per Kota, Total Transaksi per Kategori).
- Pie Chart: Gunakan HANYA untuk melihat komposisi/proporsi yang totalnya 100% (contoh: Persentase Segmen Retensi, Proporsi Gender).
- Line Chart: WAJIB gunakan jika ada kolom waktu/tanggal (order_date) untuk melihat tren.
- Scatter Plot: Gunakan untuk melihat korelasi antara dua angka (contoh: Churn Probability vs Total Spend).

CONTOH RESPON JIKA USER MINTA GRAFIK:
1. Action: query_data(sql="SELECT city, count(*) FROM customers GROUP BY city")
2. Observation: [JSON data]
3. Action: create_chart(
   data_json='[JSON DARI query_data]',
   chart_type='bar',
   title='Pelanggan per Kota',
   x_col='city',
   y_col='jumlah')
4. Final Answer: Berikut adalah grafik sebaran pelanggan Anda...
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
