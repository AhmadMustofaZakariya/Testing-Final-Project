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
    df  = pd.DataFrame(chart_data["data"])
    cols  = chart_data["columns"]
    title = chart_data["title"]
    ctype = chart_data.get("type", "bar")

    if ctype == "pie":
        fig = px.pie(df, names=cols[0], values=cols[1], title=title,
                     color_discrete_sequence=px.colors.qualitative.Set2)
    elif ctype == "line":
        fig = px.line(df, x=cols[0], y=cols[1], title=title)
    elif ctype == "scatter":
        fig = px.scatter(df, x=cols[0], y=cols[1], title=title)
    else:
        fig = px.bar(df, x=cols[0], y=cols[1], title=title,
                     color=cols[0],
                     color_discrete_sequence=px.colors.qualitative.Set2)

    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


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
if user_input:
    if not st.session_state.agent:
        st.warning("Agent belum siap, cek API key.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Menganalisis..."):
                try:
                    answer = invoke_agent(st.session_state.agent, user_input)

                    # Ambil chart dari session_state (disimpan oleh agent.py)
                    chart_data = None
                    if "CHART_READY:" in answer:
                        chart_data = st.session_state.get("pending_chart")
                        if chart_data:
                            answer = answer.replace(
                                f"CHART_READY:{chart_data['title']}", ""
                            ).strip()
                            st.session_state.pending_chart = None

                    st.markdown(answer)
                    if chart_data:
                        render_chart(chart_data)

                    # Simpan ke history
                    saved = {"role": "assistant", "content": answer}
                    if chart_data:
                        saved["chart"] = chart_data
                    st.session_state.messages.append(saved)

                except Exception as e:
                    err = f"Terjadi error: {str(e)}"
                    st.error(err)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": err}
                    )
