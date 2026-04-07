import os
import streamlit as st
import plotly.express as px
import pandas as pd
import re
import json
from agent import create_agent, invoke_agent, run_sql

# 1. Konfigurasi Halaman
st.set_page_config(page_title="AI Retail Analyst", page_icon="📊", layout="wide")

st.title("📊 AI Retail Analyst - Premium Dashboard")

# 2. Inisialisasi Agent & History
if "agent" not in st.session_state:
    st.session_state.agent = create_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. FUNGSI RENDER HISTORY (Agar grafik tidak hilang) ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Jika ada data chart tersimpan di history, gambar lagi
        if "chart_data" in msg and msg["chart_data"]:
            df_hist = pd.DataFrame(msg["chart_data"])
            if not df_hist.empty and len(df_hist.columns) >= 2:
                cols = df_hist.columns.tolist()
                fig = px.bar(df_hist, x=cols[0], y=cols[1], color_discrete_sequence=['#00CC96'])
                st.plotly_chart(fig, use_container_width=True)

# --- 4. INPUT USER ---
user_input = st.chat_input("Tanya sesuatu...")

if user_input:
    # Simpan input user
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Proses Jawaban Assistant
    with st.chat_message("assistant"):
        with st.spinner("Menganalisis data..."):
            try:
                # Ambil jawaban mentah dari LLM
                full_response = invoke_agent(st.session_state.agent, user_input)
                
                # --- LOGIKA EKSTRAKSI DATA (Anti-Gagal) ---
                # Mencari pola JSON array [ { ... } ]
                match = re.search(r'\[\s*\{.*\}\s*\]', full_response, re.DOTALL)
                
                data_for_chart = None
                display_text = full_response
                
                if match:
                    json_str = match.group(0)
                    try:
                        data_for_chart = json.loads(json_str)
                        # HAPUS JSON dari teks chat agar tidak berantakan
                        display_text = full_response.replace(json_str, "").strip()
                    except:
                        data_for_chart = None
                
                # Tampilkan teks analisis yang sudah bersih
                st.markdown(display_text)
                
                # --- LOGIKA TAMPILKAN CHART BARU ---
                if data_for_chart and len(data_for_chart) > 1:
                    df_new = pd.DataFrame(data_for_chart)
                    cols = df_new.columns.tolist()
                    
                    with st.expander("📊 Analisis Visual Otomatis", expanded=True):
                        # Pastikan kolom Y adalah angka
                        df_new[cols[1]] = pd.to_numeric(df_new[cols[1]], errors='coerce')
                        fig = px.bar(df_new, x=cols[0], y=cols[1], template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)
                
                # --- SIMPAN KE HISTORY ---
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": display_text,
                    "chart_data": data_for_chart # Simpan datanya di sini
                })
                
            except Exception as e:
                st.error(f"Terjadi kesalahan: {str(e)}")