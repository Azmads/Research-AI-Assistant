import streamlit as st
import ollama
import openai
import langchain_community
import langchain
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import pandas as pd
import PyPDF2
import pdf2image
import pytesseract
import requests
import torch
import pdf2image
import docx
from docx import Document

from transformers import AutoModelForCausalLM, AutoTokenizer

# OpenAI API key (Set in Streamlit UI)
OPENAI_API_KEY = ""

# Initialize Streamlit App
st.set_page_config(page_title="Hybrid Chatbot", page_icon="🤖", layout="wide")
st.title("💬 Hybrid AI Chatbot (Local Chat GPT-4) for Team 7 in ITAI 2376")

# Sidebar settings
st.sidebar.header("Settings")
openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
model_choice = st.sidebar.selectbox("Choose Model", ["gpt-4o", "mistral", "mixtral", "deepseek-7b"])


# Initialize OpenAI Chat Model (Only if GPT-4 is enabled)
if openai_key:
    gpt4 = ChatOpenAI(model_name="gpt-4o", temperature=0.5, openai_api_key=openai_key)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def hybrid_chatbot(prompt):
        if model_choice == "gpt-4o" and openai_key:
            response = gpt4([HumanMessage(content=prompt)])  # Enable streaming
            return f"**[GPT-4 Turbo]** {response.content}"

def process_uploaded_file(uploaded_file):
    file_type = uploaded_file.type

    if file_type == "text/plain":  # TXT file
        content = uploaded_file.read().decode("utf-8")
    elif file_type == "application/pdf":  # PDF file
        content = process_pdf(uploaded_file)
    elif file_type == "text/csv":  # CSV file
        df = pd.read_csv(uploaded_file)
        content = df.to_string()
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":  # DOCX file
        doc = Document(uploaded_file)
        content = "\n".join([para.text for para in doc.paragraphs])
    elif file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":  # Excel file
        df = pd.read_excel(uploaded_file)
        content = df.to_string()
    else:
        content = "Unsupported file format."

    return content

def summarize_excel(uploaded_file):
    """Process an Excel financial report and summarize key insights."""
    try:
        df = pd.read_excel(uploaded_file)
        summary = """
        **Excel File Summary:**
        - Number of Sheets: {sheets}
        - First Sheet Name: {first_sheet}
        - Rows: {rows}, Columns: {cols}
        """.format(sheets=len(pd.ExcelFile(uploaded_file).sheet_names),
                   first_sheet=pd.ExcelFile(uploaded_file).sheet_names[0],
                   rows=df.shape[0], cols=df.shape[1])

        # Display column names & first few rows
        summary += "\n**Columns:** " + ", ".join(df.columns) + "\n"
        summary += "\n**First 5 Rows:**\n" + df.head().to_string()
        return summary
    except Exception as e:
        return f"Error processing Excel file: {str(e)}"

def process_pdf(uploaded_file):
    """Extract text from a PDF file. Uses OCR if necessary."""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        extracted_text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])

        if not extracted_text.strip():  # If no text was extracted, use OCR
            st.warning("PDF appears to be scanned. Running OCR...")
            uploaded_file.seek(0)
            images = pdf2image.convert_from_bytes(uploaded_file.read())
            extracted_text = "\n".join([pytesseract.image_to_string(img) for img in images])

        return extracted_text
    except Exception as e:
        return f"Error processing PDF: {str(e)}"


# File Upload Button
uploaded_file = st.file_uploader("Upload a file for the agent to read", type=["txt", "pdf", "csv", "docx", "xlsx"])


user_input = st.chat_input("Type your message...")

# Process File Upload
if uploaded_file is not None:
    if uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        file_content = summarize_excel(uploaded_file)
    else:
        file_content = process_uploaded_file(uploaded_file)

    st.session_state.uploaded_file_content = file_content
    st.success("File uploaded successfully!")
    st.text_area("File Content Preview", file_content, height=200)


if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            file_content = st.session_state.get('uploaded_file_content', '').strip()

            if not file_content:
                contextual_prompt = user_input  # fallback to just the input
            else:
                contextual_prompt = f"The following is file content:\n{file_content}\n\nUser request: {user_input}"

            response = hybrid_chatbot(contextual_prompt)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})