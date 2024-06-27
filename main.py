import os
import tempfile
import streamlit as st
from rag import ChatPDF

st.set_page_config(page_title="Chat with your PDF docs")

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "assistant" not in st.session_state:
        st.session_state.assistant = ChatPDF()

def display_messages():
    st.subheader("Chat")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

def process_input(user_input):
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.assistant.ask(user_input)
                st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def read_and_save_file():
    st.session_state.assistant.clear()
    st.session_state.messages = []

    for file in st.session_state.file_uploader:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.spinner(f"Ingesting {file.name}"):
            st.session_state.assistant.ingest(file_path)
        os.remove(file_path)

def main():
    initialize_session_state()

    st.header("Chat with your PDF docs")

    st.subheader("Upload a document")
    st.file_uploader(
        "Upload document",
        type=["pdf"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    display_messages()

    user_input = st.chat_input("Type your message here...")
    if user_input:
        process_input(user_input)

if __name__ == "__main__":
    main()
