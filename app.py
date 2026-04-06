"""
app.py
Streamlit Chat UI untuk AI Retail Analyst
"""

import os
import json
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from dummy_db import create_and_populate
from agent import create_agent

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="AI Retail Analyst",
    page_icon="📊",
    layout="wide"
)

# -------------------------------------------------------
# INIT: buat dummy DB kalau belum ada
# -------------------------------------------------------
if not os.path.exists("retail_dummy.db"):
    create_and_populate()

# -------------------------------------------------------
# SESSION STATE
# -------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None

# -------------------------------------------------------
# SIDEBAR: API Key input
# -------------------------------------------------------
with st.sidebar:
    st.title("📊 AI Retail Analyst")
    st.caption("Powered by Gemini AI")
    st.divider()

    # Cek API key dari environment
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        if st.session_state.agent is None:
            st.session_state.agent = create_agent()
        st.success("Agent siap digunakan!")
    else:
        st.error("GEMINI_API_KEY tidak ditemukan!")
        st.caption("Pastikan .env sudah diisi")

    st.divider()


    # Contoh pertanyaan
    st.subheader("💡 Contoh pertanyaan")
    example_questions = [
        "Berapa jumlah pelanggan high risk churn?",
        "Tampilkan distribusi segmen retensi dalam grafik",
        "Kota mana yang memiliki churn tertinggi?",
        "Siapa 5 pelanggan dengan churn probability tertinggi?",
        "Berapa rata-rata churn probability pelanggan wanita?",
        "Tampilkan chart churn per kota",
    ]
    for q in example_questions:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q

# -------------------------------------------------------
# MAIN: Chat Interface
# -------------------------------------------------------
st.title("📊 AI Retail Analyst")
st.caption("Tanyakan apapun tentang data pelanggan, churn, dan retensi")

# Tampilkan chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Render chart kalau ada
        if "chart" in msg:
            render_chart(msg["chart"])

# Helper: render chart dari data JSON
def render_chart(chart_data: dict):
    df = pd.DataFrame(chart_data["data"])
    cols = chart_data["columns"]
    title = chart_data["title"]

    if chart_data["type"] == "pie" and len(cols) >= 2:
        fig = px.pie(df, names=cols[0], values=cols[1], title=title)
    elif chart_data["type"] == "bar" and len(cols) >= 2:
        fig = px.bar(df, x=cols[0], y=cols[1], title=title,
                     color=cols[0], color_discrete_sequence=px.colors.qualitative.Set2)
    else:
        fig = px.bar(df, x=cols[0], y=cols[1], title=title)

    fig.update_layout(showlegend=True, height=400)
    st.plotly_chart(fig, use_container_width=True)


# Handle pertanyaan dari sidebar button
if "pending_question" in st.session_state:
    user_input = st.session_state.pending_question
    del st.session_state.pending_question
else:
    user_input = st.chat_input("Tanyakan sesuatu tentang data pelanggan...")

# Process input
if user_input:
    if not st.session_state.agent:
        st.warning("Masukkan Groq API Key di sidebar terlebih dahulu.")
    else:
        # Tampilkan pesan user
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Jalankan agent
        with st.chat_message("assistant"):
            with st.spinner("Menganalisis..."):
                try:
                    from agent import invoke_agent
                    answer = invoke_agent(st.session_state.agent, user_input)


                    # Cek apakah ada chart yang perlu dirender
                    chart_data = None
                    if "CHART_READY:" in answer and os.path.exists("chart_data.json"):
                        with open("chart_data.json") as f:
                            chart_data = json.load(f)
                        # Bersihkan marker dari teks jawaban
                        answer = answer.replace(
                            f"CHART_READY:{chart_data['title']}", ""
                        ).strip()

                    st.markdown(answer)
                    if chart_data:
                        render_chart(chart_data)

                    # Simpan ke history
                    msg = {"role": "assistant", "content": answer}
                    if chart_data:
                        msg["chart"] = chart_data
                    st.session_state.messages.append(msg)

                except Exception as e:
                    err = f"Terjadi error: {str(e)}"
                    st.error(err)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": err}
                    )
