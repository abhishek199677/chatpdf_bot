import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(text):
    if not text:
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    if not text_chunks:
        raise ValueError("No text chunks available for vector storage")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible and creatively from the provided context only,
    make sure to provide all the details. If the answer isn't in the context, say 
    "I couldn't find that information in the documents." \n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.4)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings)
        docs = new_db.similarity_search(user_question)
        
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        return response["output_text"]
    except Exception as e:
        return "Please process PDF documents first before asking questions."

def main():
    st.set_page_config("Chat PDF", page_icon="📄")
    st.header("Multiple PDF Q&A with AI 🤖")

    user_question = st.text_input("Ask a Question from the PDF Files")
    
    if user_question:
        response = user_input(user_question)
        st.write("Reply:")
        st.info(response)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload PDF Files (Max 5)",
            accept_multiple_files=True,
            type="pdf"
        )
        
        if st.button("Process PDFs"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
                return
                
            with st.spinner("Processing..."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text:
                        st.error("No text could be extracted from the uploaded PDFs.")
                        return
                        
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("PDFs processed successfully!")
                except Exception as e:
                    st.error(f"Error processing PDFs: {str(e)}")

if __name__ == "__main__":
    main()

st.markdown("""
    <style>
        .faded-text {
            opacity: 0.5;
            font-size: 0.8em;
            margin-top: 2rem;
        }
    </style>
    <div class="faded-text">
        Made by Abhishek
    </div>
""", unsafe_allow_html=True)

