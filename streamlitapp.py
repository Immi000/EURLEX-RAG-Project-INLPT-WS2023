import streamlit as sl
from chatmodel import *

sl.set_page_config(layout="wide")

if "QA_instance" not in sl.session_state:
    QA_instance = RAG(verbose=True)
    sl.session_state.QA_instance = QA_instance

default_message = {
    "role": "bot",
    "content": "Welcome to the EUR-Lex Chatbot! Feel free to ask any questions around rulings by the european union regarding energy law.",
}


def clear():
    sl.session_state.QA_instance.clear_history()
    sl.session_state.messages = []
    sl.session_state.messages.append(default_message)


col1, col2, col3 = sl.columns([6, 3, 1])
with col1:
    sl.title("EU Energy Law Chatbot")
with col2:
    option = sl.selectbox(
        "Index",
        (tuple(list(INDEX_MAP.keys()))),
        index=0,
    )
    if sl.session_state.QA_instance.index_name != option:
        sl.session_state.QA_instance.set_index(option)
with col3:
    sl.button("Clear History", on_click=clear)

if "messages" not in sl.session_state:
    sl.session_state.messages = []
    sl.session_state.messages.append(default_message)

for message in sl.session_state.messages:
    with sl.chat_message(message["role"]):
        sl.markdown(message["content"])

prompt = sl.chat_input("Please enter your query", key="prompt")
if prompt:
    with sl.chat_message("user"):
        sl.markdown(prompt)
    sl.session_state.messages.append({"role": "user", "content": prompt})

    response = sl.session_state.QA_instance.ask_question(prompt)

    with sl.chat_message("bot"):
        if response.startswith("Helpful answer:"):
            response = response.replace("Helpful answer:", "", 1)
        if response.startswith("**Helpful answer:**"):
            response = response.replace("**Helpful answer:**", "", 1)
        sl.markdown(response)
    sl.session_state.messages.append({"role": "bot", "content": response})
