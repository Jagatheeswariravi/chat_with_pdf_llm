import streamlit as st
from dotenv import load_dotenv
import fitz
from io import BytesIO
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import ctransformers,huggingface_hub
from langchain.vectorstores import Pinecone
from langchain.vectorstores import FAISS
from pydantic import BaseModel
from htmlTemplate import css,bot_template,user_template
import pinecone

OPENAI_API_TOKEN = ""
HUGGINGFACEHUB_API_TAKEN="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"
PINECONE_API_KEY = "37e975e8-17d7-4e8e-a98d-9dd093cf133f"
PINECONE_API_ENV = "gcp-starter"

import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


def text_from_pdf(pdf_docs):
    text = ""
    for pdf_doc in pdf_docs:
        # Open each PDF using fitz.open
        doc = fitz.open(stream=BytesIO(pdf_doc.read()))
        
        for page_number in range(doc.page_count):
            page = doc[page_number]
            text += page.get_text()

        doc.close()

    return text

def text_to_chunks(raw_text):
    spliter = CharacterTextSplitter(separator="\n",chunk_size=500,chunk_overlap =100,length_function=len)
    chunks = spliter.split_text(raw_text)
    return chunks


def chunks_to_embeds(chunks,pdf_docs):
    
# initialize pinecone
    pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
    index_name = "pdfreader"
    try:
        
        embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    except Exception as e:
        st.error(f"Error during embeddings initialization: {str(e)}")
        return None

    try:
        vector_store = Pinecone.from_documents(pdf_docs,embeddings, index_name=index_name)
     
    except Exception as e:
        st.error(f"Error during vector store creation: {str(e)}")
        return None

    return vector_store

def get_conversation_chain(vector_store):

    llm = huggingface_hub.HuggingFaceHub(repo_id = "google/flan-t5-xxl",
                        model_kwargs = { "temperature":0.5,"max_length" :512})

    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm = llm,retriever=vector_store,memory=memory)

    return conversation_chain

def handle_user_question(user_question):
    response = st.session_state.conversation({"question":user_question})
    st.session_state.chat_history = response["chat_history"]

    for i,message in enumerate(st.session_state.chat_history):
        if i%2 ==0:
            st.write(user_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title = "chat with pdf")
    st.write(css,unsafe_allow_html=True)
    
    if "conversion" not in st.session_state:
        st.session_state.conversation = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("CHAT WITH PDF")
    user_question = st.text_input("Ask a question about your document")

    if user_question:
        handle_user_question(user_question)



    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs =st.file_uploader("Upload your pdf and click process",accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = text_from_pdf(pdf_docs)
                chunks = text_to_chunks(raw_text)
                vector_store = chunks_to_embeds(chunks,pdf_docs)
                st.session_state.conversation = get_conversation_chain(vector_store)


 


if __name__ == "__main__":
    main()