"""
app.py - CLEAN VERSION
Visualisasi dihandle app.py langsung — tidak bergantung pada LLM untuk chart.
"""

import os
import streamlit as st
import plotly.express as px
import pandas as pd
from dummy_db import create_and_populate
from agent import create_agent, invoke_agent, run_sql
import re
import json

st.set_page_config(page_title="AI Retail Analyst", page_icon="📊", layout="wide")

# -------------------------------------------------------
# CHART CONFIG
# Mapping keyword → SQL + chart config
# Ini yang bikin chart selalu muncul tanpa bergantung LLM
# -------------------------------------------------------
CHART_CONFIG = [
    {
        "keywords": ["segmen retensi", "retention segment", "segmen pelanggan"],
        "sql": "SELECT retention_segment, COUNT(*) as jumlah FROM predictions GROUP BY retention_segment ORDER BY jumlah DESC",
        "type": "pie",
        "title": "Distribusi Segmen Retensi Pelanggan",
        "x": "retention_segment",
        "y": "jumlah"
    },
    {
        "keywords": ["churn per kota", "churn kota", "kota churn", "churn tertinggi"],
        "sql": """SELECT c.city, ROUND(AVG(p.churn_probability),2) as avg_churn
                  FROM predictions p JOIN customers c ON p.customer_id = c.customer_id
                  GROUP BY c.city ORDER BY avg_churn DESC""",
        "type": "bar",
        "title": "Rata-rata Churn Probability per Kota",
        "x": "city",
        "y": "avg_churn"
    },
    {
        "keywords": ["distribusi churn", "churn risk", "high risk", "medium risk", "low risk"],
        "sql": "SELECT churn_label, COUNT(*) as jumlah FROM predictions GROUP BY churn_label ORDER BY jumlah DESC",
        "type": "bar",
        "title": "Distribusi Churn Risk Pelanggan",
        "x": "churn_label",
        "y": "jumlah"
    },
    {
        "keywords": ["kategori produk", "product category", "kategori"],
        "sql": """SELECT product_category, COUNT(*) as total_transaksi
                  FROM transactions GROUP BY product_category ORDER BY total_transaksi DESC""",
        "type": "bar",
        "title": "Transaksi per Kategori Produk",
        "x": "product_category",
        "y": "total_transaksi"
    },
    {
        "keywords": ["churn gender", "gender churn", "wanita", "pria", "gender"],
        "sql": """SELECT c.gender, ROUND(AVG(p.churn_probability),2) as avg_churn
                  FROM predictions p JOIN customers c ON p.customer_id = c.customer_id
                  GROUP BY c.gender""",
        "type": "bar",
        "title": "Rata-rata Churn Probability per Gender",
        "x": "gender",
        "y": "avg_churn"
    },
]


def get_chart_config(user_input: str):
    """Cari chart config yang cocok berdasarkan keyword di pertanyaan user."""
    user_lower = user_input.lower()
    # Cek dulu apakah user minta visualisasi
    viz_keywords = ["grafik", "chart", "visualisasi", "diagram", "plot", "tampilkan", "gambar"]
    wants_chart = any(k in user_lower for k in viz_keywords)
    if not wants_chart:
        return None
    # Cari config yang cocok
    for config in CHART_CONFIG:
        if any(k in user_lower for k in config["keywords"]):
            return config
    # Default: tampilkan distribusi churn kalau tidak ada yang cocok
    return CHART_CONFIG[2]


def render_chart(chart_data: dict):
    """Render chart dari dict config."""
    df = pd.DataFrame(chart_data["data"])
    x, y = chart_data["x"], chart_data["y"]
    df[y] = pd.to_numeric(df[y], errors="coerce")
    title = chart_data["title"]

    if chart_data["type"] == "pie":
        fig = px.pie(df, names=x, values=y, title=title, hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Set2)
    elif chart_data["type"] == "line":
        fig = px.line(df, x=x, y=y, title=title, markers=True)
    else:
        df = df.sort_values(by=y, ascending=False)
        fig = px.bar(df, x=x, y=y, title=title, color=x,
                     color_discrete_sequence=px.colors.qualitative.Set2)

    fig.update_layout(height=420, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------
# INIT
# -------------------------------------------------------
if not os.path.exists("retail_dummy.db"):
    create_and_populate()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------
with st.sidebar:
    st.title("📊 AI Retail Analyst")
    st.caption("Powered by Groq + LLaMA")
    st.divider()

    if os.getenv("GROQ_API_KEY"):
        if st.session_state.agent is None:
            st.session_state.agent = create_agent()
        st.success("Agent siap digunakan!")
    else:
        st.error("GROQ_API_KEY tidak ditemukan!")

    st.divider()
    st.subheader("💡 Contoh pertanyaan")
    questions = [
        "Berapa jumlah pelanggan high risk churn?",
        "Tampilkan grafik distribusi segmen retensi",
        "Tampilkan chart churn per kota",
        "Siapa 5 pelanggan dengan churn probability tertinggi?",
        "Tampilkan grafik distribusi churn risk",
        "Tampilkan chart churn per gender",
    ]
    for q in questions:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q

    if st.button("🗑️ Reset Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
st.title("📊 AI Retail Analyst")
st.caption("Tanyakan apapun tentang data pelanggan, churn, dan retensi")

# Render history
# --- 1. BAGIAN MENAMPILKAN HISTORY (Agar chart lama gak hilang) ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Gambar ulang grafik dari history jika ada data_chart
        if "chart_data" in msg and msg["chart_data"]:
            df_hist = pd.DataFrame(msg["chart_data"])
            cols = df_hist.columns.tolist()
            if len(cols) >= 2:
                fig = px.bar(df_hist, x=cols[0], y=cols[1])
                st.plotly_chart(fig, use_container_width=True)

# Chat input
user_input = st.chat_input("Tanyakan sesuatu tentang data pelanggan...")
if "pending_question" in st.session_state:
    user_input = st.session_state.pop("pending_question")

# -------------------------------------------------------
# PROCESS
# -------------------------------------------------------
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Menganalisis..."):
            # 1. Ambil jawaban dari LLM
            full_response = invoke_agent(st.session_state.agent, user_input)
            
            # 2. Pisahkan Teks Analisis dan JSON (jika ada)
            # Regex ini buat nyari JSON array [ { ... } ]
            match = re.search(r'\[\s*\{.*\}\s*\]', full_response, re.DOTALL)
            
            data_list = None
            clean_answer = full_response
            
            if match:
                try:
                    json_str = match.group(0)
                    data_list = json.loads(json_str)
                    # Buang JSON-nya dari teks biar chat bersih
                    clean_answer = full_response.replace(json_str, "").strip()
                except:
                    pass

            # 3. Tampilkan teks analisis (yang sudah bersih dari JSON)
            st.markdown(clean_answer)

            # 4. LOGIKA VISUALISASI (Hybrid)
            chart_to_save = None
            
            # Cek cara Claude dulu (Template)
            chart_config = get_chart_config(user_input)
            
            if chart_config:
                # Jika pertanyaan sesuai template, jalankan SQL Claude
                df_sql = run_sql(chart_config["sql"])
                if not df_sql.empty:
                    chart_to_save = df_sql.to_dict(orient="records")
                    # Tampilkan grafik cara Claude
                    fig = px.bar(df_sql, x=chart_config["x"], y=chart_config["y"], title=chart_config["title"])
                    st.plotly_chart(fig, use_container_width=True)
            
            elif data_list and len(data_list) > 1:
                # Jika BUKAN template, tapi LLM pinter ngasih data (Dinamis)
                df_dyn = pd.DataFrame(data_list)
                cols = df_dyn.columns.tolist()
                chart_to_save = data_list
                with st.expander("📊 Analisis Visual Otomatis", expanded=True):
                    fig = px.bar(df_dyn, x=cols[0], y=cols[1], title="Visualisasi Data")
                    st.plotly_chart(fig, use_container_width=True)

            # 5. SIMPAN KE HISTORY (Biar gak hilang pas scroll)
            st.session_state.messages.append({
                "role": "assistant",
                "content": clean_answer,
                "chart_data": chart_to_save
            })
