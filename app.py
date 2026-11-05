import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from main import query

load_dotenv()

st.title("🤖 Chatbot Q&A App")
chat_placeholder = st.empty()


def init_chat_history():
    """Initialize chat history with a system message."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append(
            SystemMessage(content="You are a helpful robot assistant.")
        )


def _role_for(message):
    """Map a LangChain message to a Streamlit chat role."""
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    return "system"


def start_chat():
    """Start the chatbot conversation."""
    # Replay full history on every rerun (skip the system prompt).
    with chat_placeholder.container():
        for message in st.session_state.messages:
            if isinstance(message, SystemMessage):
                continue
            with st.chat_message(_role_for(message)):
                st.markdown(message.content)

    if prompt := st.chat_input("Ask a question:"):
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        response = query(prompt)["answer"]
        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append(AIMessage(content=response))


if __name__ == "__main__":
    init_chat_history()
    start_chat()
