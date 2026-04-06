"""
test_agent.py
Test agent langsung di terminal tanpa perlu Streamlit.
Jalankan: python test_agent.py
"""

from dummy_db import create_and_populate
from dotenv import load_dotenv
from agent import create_agent, invoke_agent
import os

load_dotenv()

if not os.path.exists("retail_dummy.db"):
    print("Creating dummy database...")
    create_and_populate()
    print()

GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxx"  # ganti dengan key kamu

print("Initializing agent...")
agent = create_agent(GROQ_API_KEY)
print("Agent ready!\n")
print("=" * 50)

test_questions = [
    "Berapa jumlah pelanggan yang high risk churn?",
    "Kota mana yang memiliki rata-rata churn probability tertinggi?",
    "Tampilkan 5 pelanggan dengan churn probability tertinggi",
    "Berapa jumlah pelanggan di setiap segmen retensi?",
    "Berapa rata-rata churn probability pelanggan wanita vs pria?",
    "Tampilkan grafik distribusi segmen retensi pelanggan",
]

question = test_questions[0]
print(f"Question: {question}")
print("-" * 50)
answer = invoke_agent(agent, question)
print(f"\nAnswer:\n{answer}")
print("=" * 50)

# Uncomment untuk interactive mode:
# print("\nInteractive mode (ketik 'exit' untuk keluar):")
# while True:
#     user_input = input("\nKamu: ").strip()
#     if user_input.lower() == "exit":
#         break
#     if not user_input:
#         continue
#     print(f"\nAgent: {invoke_agent(agent, user_input)}")

