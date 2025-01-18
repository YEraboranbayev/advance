import streamlit as st
import logging
import time
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
import PyPDF2

logging.basicConfig(level=logging.INFO)

def read_file(file):
    if file.name.endswith('.pdf'):
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            logging.error(f"Error reading PDF: {str(e)}")
            return "Failed to read PDF file."
    elif file.name.endswith('.txt'):
        try:
            return file.read().decode("utf-8")
        except Exception as e:
            logging.error(f"Error reading TXT: {str(e)}")
            return "Failed to read text file."
    else:
        return "Unsupported file format. Please upload a PDF or TXT file."

def stream_chat(model, messages, temperature=0.7, max_tokens=256):
    try:
        llm = Ollama(model=model, request_timeout=120.0)
        response = ""
        response_placeholder = st.empty()

        resp = llm.stream_chat(messages, temperature=temperature, max_tokens=max_tokens)
        for r in resp:
            response += r.delta
            response_placeholder.write(response)

        logging.info(f"Model: {model}, Messages: {messages}, Response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error during streaming: {str(e)}")
        raise e

def main():
    st.title("AI Assistant")
    logging.info("Application started")

    model = st.sidebar.selectbox("Select model", ["llama3.1:8b"])
    logging.info(f"Model selected: {model}")

    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "You are a smart assistant analyzing the Constitution of Kazakhstan."}
        ]

    uploaded_file = st.sidebar.file_uploader("Upload Constitution file (PDF or TXT)", type=["pdf", "txt"])
    constitution_text = ""

    if uploaded_file:
        file_content = read_file(uploaded_file)
        if file_content.startswith("Unsupported file format"):
            st.sidebar.error(file_content)
        else:
            st.sidebar.success("File successfully uploaded.")
            constitution_text = file_content
            st.session_state.messages.append({"role": "system", "content": "The Constitution text has been loaded."})

    if prompt := st.chat_input("Enter your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        logging.info(f"User input: {prompt}")

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            start_time = time.time()
            logging.info("Generating response...")

            with st.spinner("Generating response..."):
                try:
                    messages = [ChatMessage(role=msg["role"], content=msg["content"]) for msg in st.session_state.messages]
                    response_message = stream_chat(model, messages)
                    duration = time.time() - start_time

                    st.session_state.messages.append({"role": "assistant", "content": response_message})
                    st.write(f"Answer: {response_message} (generated in {duration:.2f} seconds)")
                    logging.info(f"Answer: {response_message}, Time: {duration:.2f} seconds")
                except Exception as e:
                    st.session_state.messages.append({"role": "assistant", "content": str(e)})
                    st.error("Error generating response.")
                    logging.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
