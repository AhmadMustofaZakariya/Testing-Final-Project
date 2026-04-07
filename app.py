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
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "chart" in msg:
            render_chart(msg["chart"])

# Chat input
user_input = st.chat_input("Tanyakan sesuatu tentang data pelanggan...")
if "pending_question" in st.session_state:
    user_input = st.session_state.pop("pending_question")

# -------------------------------------------------------
# PROCESS
# -------------------------------------------------------
if user_input:
    if not st.session_state.agent:
        st.warning("Agent belum siap, cek API key.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Menganalisis data retail..."):
                # 1. Panggil Agent (Llama 4 Scout)
                full_response = invoke_agent(st.session_state.agent, user_input)
                
                # 2. DETEKSI DINAMIS (Cari JSON di jawaban LLM)
                import re
                import json
                
                # Regex untuk menangkap pola JSON array [ {...} ]
                match = re.search(r'\[\s*\{.*\}\s*\]', full_response, re.DOTALL)
                
                data_list = None
                clean_answer = full_response
                
                if match:
                    try:
                        json_str = match.group(0)
                        data_list = json.loads(json_str)
                        # Hapus JSON mentah dari chat agar UI tetap bersih & profesional
                        clean_answer = full_response.replace(json_str, "").strip()
                    except:
                        pass # Jika gagal parse, biarkan saja

                # 3. Tampilkan Jawaban Teks (Jawaban Llama 4 yang sudah bersih)
                st.markdown(clean_answer)

                # 4. TAMPILKAN GRAFIK
                # Cek dulu: Apakah ini pertanyaan yang ada di CHART_CONFIG (Cara Claude)?
                chart_config = get_chart_config(user_input)
                
                if chart_config:
                    # Jika ada di config Claude, jalankan cara Claude
                    df_c = run_sql(chart_config["sql"])
                    if not df_c.empty:
                        render_chart({
                            "type": chart_config["type"],
                            "title": chart_config["title"],
                            "x": chart_config["x"],
                            "y": chart_config["y"],
                            "data": df_c.to_dict(orient="records")
                        })
                elif data_list and len(data_list) > 1:
                    # Jika GAK ADA di config Claude, tapi LLM ngasih data JSON (Cara Dinamis)
                    df_d = pd.DataFrame(data_list)
                    cols = df_d.columns.tolist()
                    
                    with st.expander("📊 Analisis Visual Otomatis", expanded=True):
                        # Pastikan kolom kedua adalah angka
                        df_d[cols[1]] = pd.to_numeric(df_d[cols[1]], errors='coerce')
                        
                        fig = px.bar(df_d, x=cols[0], y=cols[1], 
                                    title=f"Trend {cols[1]} per {cols[0]}",
                                    color_discrete_sequence=['#00CC96'])
                        st.plotly_chart(fig, use_container_width=True)

                # 5. Simpan ke History (PENTING agar grafik tidak hilang saat scroll)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": clean_answer,
                    "data": data_list # Simpan datanya di sini
                })
