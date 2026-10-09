import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenAIEmbeddings, ChatGoogleGenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Page Configuration
st.set_page_config(page_title="Instant Document Q&A", page_icon="📄", layout="centered")
st.title("📄 AI-Powered Document Q&A (RAG)")
st.write("Upload a PDF and ask questions about its content instantly.")

# 2. API Key Input Securely in Sidebar
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter Google Gemini API Key:", type="password")
    st.markdown("[Get a free key here](https://google.com)")

if not api_key:
    st.info("Please enter your Gemini API key in the sidebar to begin.", icon="🔑")
    st.stop()

# 3. File Uploader UI
uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")

if uploaded_file is not None:
    with st.spinner("Processing document... This takes only a few seconds."):
        # Step A: Extract text from PDF
        pdf_reader = PdfReader(uploaded_file)
        raw_text = ""
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text
        
        # Step B: Split text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        text_chunks = text_splitter.split_text(raw_text)
        
        # Step C: Generate Embeddings and save to a local FAISS Vector Store
        embeddings = GoogleGenAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        
    st.success("Document processed successfully! Ask your questions below.")

    # 4. Q&A Interface
    user_question = st.text_input("What would you like to know from this document?")
    
    if user_question:
        with st.spinner("Analyzing document and generating answer..."):
            # Step D: Define the LLM and the RAG System Prompt
            llm = ChatGoogleGenAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.3)
            
            system_prompt = (
                "You are an assistant for text-based question-answering tasks. "
                "Use the following pieces of retrieved context to answer the question. "
                "If you don't know the answer, say that you don't know.\n\n"
                "Context:\n{context}"
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            
            # Step E: Combine components into a RAG Chain
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            
            # Step F: Run the query
            response = rag_chain.invoke({"input": user_question})
            
            # Step G: Output Answer
            st.markdown("### 🤖 Answer:")
            st.write(response["answer"])
