"""
agent.py - CLEAN VERSION
LLM hanya bertugas sebagai Text-to-SQL dan menjawab pertanyaan.
Visualisasi dihandle sepenuhnya oleh app.py
"""

import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv()

DB_PATH = "retail_dummy.db"

SCHEMA_INFO = """
Database memiliki 3 tabel:

1. customers (customer_id, age, gender, city, is_returning)
2. transactions (order_id, customer_id, order_date, product_category,
                 total_amount, payment_method, device_type,
                 customer_rating, delivery_days)
3. predictions (customer_id, churn_probability, churn_label,
                retention_segment, recency_days, frequency,
                monetary, predicted_at)
   - churn_label: 'High Risk', 'Medium Risk', 'Low Risk'
   - retention_segment: 'Champions', 'Loyal', 'At Risk', 'Lost'

Untuk churn/retensi gunakan tabel predictions.
JOIN customers jika butuh city/gender/age.
JOIN transactions jika butuh product_category/total_amount.
"""

SYSTEM_PROMPT = """Kamu adalah AI Retail Analyst untuk perusahaan e-commerce.
Tugasmu membantu stakeholder (CMO, CEO) memahami data pelanggan
terkait churn dan retensi.

INSTRUKSI PENTING:
1. Berikan analisis bisnis yang tajam dalam Bahasa Indonesia.
2. Setiap kali kamu mengambil data dari SQL, kamu WAJIB menuliskan data tersebut dalam format JSON array di baris paling bawah jawabanmu.
3. Gunakan format: [{"label": "Nama", "value": 123}, ...]
4. JANGAN gunakan markdown code block (```json). Tulis saja mentah-mentah di baris terakhir.

{SCHEMA_INFO}

Aturan:
- SELALU gunakan tool query_data untuk mengambil data
- Jawab dalam Bahasa Indonesia yang mudah dipahami
- Berikan interpretasi bisnis dari data yang ditemukan
- Jangan tampilkan SQL mentah ke user
- Jika data kosong, jelaskan dengan sopan
"""


def run_sql(query: str) -> pd.DataFrame:
    """Helper publik: jalankan SQL, return DataFrame"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


@tool
def query_data(sql: str) -> str:
    """Gunakan tool ini untuk mengambil data dari database.
    Input: SQL query yang valid.
    Gunakan untuk semua pertanyaan yang butuh data."""
    try:
        df = run_sql(sql)
        if df.empty:
            return "Query berhasil tapi tidak ada data."
        return df.head(20).to_string(index=False)
    except Exception as e:
        return f"SQL Error: {str(e)}"


def create_agent():
    llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )
    tools = [query_data]
    agent = create_react_agent(llm, tools, state_modifier=SYSTEM_PROMPT)
    return agent


def invoke_agent(agent, user_input: str) -> str:
    result = agent.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config={"recursion_limit": 25}
    )
    return result["messages"][-1].content
