"""
app.py - FIXED VERSION
Streamlit Chat UI untuk AI Retail Analyst

Fix:
1. render_chart() didefinisikan di atas sebelum dipanggil
2. st.chat_input() selalu ada di main flow
3. Chart data disimpan di st.session_state, bukan file JSON
"""

import os
import streamlit as st
import plotly.express as px
import pandas as pd
from dummy_db import create_and_populate
from agent import create_agent, invoke_agent

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="AI Retail Analyst",
    page_icon="📊",
    layout="wide"
)

# -------------------------------------------------------
# HELPER: render chart
# Didefinisikan PALING ATAS supaya bisa dipanggil di mana saja
# -------------------------------------------------------
def render_chart(chart_data: dict):
    # DEBUG: Hapus ini kalau sudah jalan
    st.write("Data masuk ke render_chart:", chart_data) 

    # Validasi ketat
    if not chart_data or "data" not in chart_data or "columns" not in chart_data:
        return # Diam saja, jangan tampilkan warning agar UI tidak kotor

    df = pd.DataFrame(chart_data["data"])
    cols = chart_data["columns"]
    # Kadang kolom sumbu Y (angka) terbaca sebagai string, kita paksa jadi numeric
    df[cols[1]] = pd.to_numeric(df[cols[1]], errors='coerce')
    title = chart_data["title"]
    ctype = chart_data.get("type", "bar")

    # Pastikan data angka benar-benar bertipe numeric agar grafik tidak berantakan
    if len(cols) > 1:
        df[cols[1]] = pd.to_numeric(df[cols[1]], errors='coerce')

    if ctype == "pie":
        fig = px.pie(df, names=cols[0], values=cols[1], title=title,
                     hole=0.4, # Bikin jadi Donut Chart agar lebih modern
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    elif ctype == "line":
        fig = px.line(df, x=cols[0], y=cols[1], title=title, markers=True)
    elif ctype == "scatter":
        fig = px.scatter(df, x=cols[0], y=cols[1], title=title, 
                         trendline="ols") # Tambahkan garis tren otomatis
    else:
        # Sort data agar Bar Chart rapi dari terbesar ke terkecil
        df = df.sort_values(by=cols[1], ascending=False)
        fig = px.bar(df, x=cols[0], y=cols[1], title=title,
                     color=cols[1], # Warna gradasi berdasarkan nilai
                     color_continuous_scale='Viridis')

    fig.update_layout(hovermode="x unified", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# HELPER: Fungsi untuk mencari JSON di dalam teks jawaban LLM
def extract_data_from_text(text):
    try:
        # Cari pola [ { ... } ] di dalam teks
        match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
    except:
        return None
    return None

# -------------------------------------------------------
# INIT DB
# -------------------------------------------------------
if not os.path.exists("retail_dummy.db"):
    create_and_populate()

# -------------------------------------------------------
# SESSION STATE
# -------------------------------------------------------
if "messages"     not in st.session_state:
    st.session_state.messages     = []
if "agent"        not in st.session_state:
    st.session_state.agent        = None
if "pending_chart" not in st.session_state:
    st.session_state.pending_chart = None  # ← chart disimpan di sini

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------
if st.button("Reset Chat"):
    st.session_state.messages = []
    st.session_state.pending_chart = None
    st.rerun()

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
        st.caption("Pastikan Secrets sudah diisi")

    st.divider()
    st.subheader("💡 Contoh pertanyaan")
    questions = [
        "Berapa jumlah pelanggan high risk churn?",
        "Tampilkan distribusi segmen retensi dalam grafik",
        "Kota mana yang memiliki churn tertinggi?",
        "Siapa 5 pelanggan dengan churn probability tertinggi?",
        "Berapa rata-rata churn probability pelanggan wanita?",
        "Tampilkan chart churn per kota",
    ]
    for q in questions:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q

# -------------------------------------------------------
# MAIN: Chat Interface
# -------------------------------------------------------
st.title("📊 AI Retail Analyst")
st.caption("Tanyakan apapun tentang data pelanggan, churn, dan retensi")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "chart" in msg:
            render_chart(msg["chart"])

# -------------------------------------------------------
# CHAT INPUT — harus selalu ada, tidak boleh di dalam kondisi
# -------------------------------------------------------
user_input = st.chat_input("Tanyakan sesuatu tentang data pelanggan...")

# Override dengan sidebar button jika ada
if "pending_question" in st.session_state:
    user_input = st.session_state.pop("pending_question")

# -------------------------------------------------------
# PROCESS
# -------------------------------------------------------
# DI APP.PY - Pastikan blok ini sejajar di kiri (tidak masuk ke dalam blok lain)
if user_input:
    # 1. SIMPAN PESAN USER KE HISTORY (Wajib Pertama!)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Langsung tampilkan di UI supaya user tahu input diterima
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. PROSES JAWABAN ASSISTANT
    with st.chat_message("assistant"):
        with st.spinner("Menganalisis data retail..."):
            answer = invoke_agent(st.session_state.agent, user_input)
            
            # 1. Coba ekstrak data dari teks jawaban
            data_list = extract_data_from_text(answer)
            
            # 2. Bersihkan teks dari blok JSON mentah (biar gak menuh-menuhin chat)
            clean_answer = re.sub(r'\[\s*\{.*\}\s*\]', '', answer, flags=re.DOTALL).strip()
            st.markdown(clean_answer)
            
            # 3. AUTO-CHART: Kalau ada data dan lebih dari 1 baris, tampilin grafik!
            if data_list and len(data_list) > 1:
                df = pd.DataFrame(data_list)
                
                # Kita buat expander biar UI tetep rapi
                with st.expander("📊 Visualisasi Otomatis", expanded=True):
                    # Ambil kolom pertama sebagai X, kolom kedua sebagai Y
                    cols = df.columns.tolist()
                    if len(cols) >= 2:
                        fig = px.bar(df, x=cols[0], y=cols[1], title="Analisis Data")
                        st.plotly_chart(fig, use_container_width=True)
            
            # 4. Simpan ke history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": clean_answer,
                "data": data_list # Simpan datanya biar gak ilang pas scroll
            })