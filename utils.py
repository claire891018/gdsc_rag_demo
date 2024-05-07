# 套件導入
import streamlit as st
import tempfile
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate

import main as m

google_api_key = m.google_api_key
file_pdf = m.file_pdf

def google_llm(google_api_key):
    llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key)
    return llm

def query_llm(retriever, query):
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=google_llm(google_api_key),
        retriever=retriever,
        return_source_documents=True,
    )
    result = qa_chain({'question': query, 'chat_history': st.session_state.messages})
    result = result['answer']
    st.session_state.messages.append((query, result))
    return result

def query_llm_direct(query):
    llm_chain = add_prompt(google_llm(google_api_key), query)
    result = llm_chain.invoke({"query": query})
    result = result['text']
    st.session_state.messages.append((query, result))
    return result

def add_prompt(llm, query):
    init_Prompt = """
    資料：{context}
    你能協助閱讀資料，並且總是能夠立即、準確地回答任何要求。
    請用繁體中文(traditonal chinese)回答下列問題。不知道就回答「資訊未提及：{query}」，並確保答案內容相關且簡潔。
    問題：{query}
    回答：
    """
    input_prompt = PromptTemplate(input_variables=["query"], template=init_Prompt)
    return LLMChain(prompt=input_prompt, llm=llm)

def input_fields():
    st.session_state.source_docs = st.file_uploader(label="上傳文件！", type="pdf", accept_multiple_files=True)

def load_and_chunk(file_path):
    loader = DirectoryLoader(file_path, glob='**/*.pdf')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
    texts = text_splitter.split_documents(documents)
    return texts

def create_embeddings_and_vectordb(texts):
    model_name = "aspire/acge_text_embedding"
    model_kwargs = {'device': 'cpu'}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    vectordb = Chroma.from_documents(texts, embedding=embeddings)
    retriver = vectordb.as_retriever(search_kwargs={'k': 7})
    return retriver

# 處理文件
def process_documents():
    if not google_api_key:
        st.warning("請上傳你的 api key！")
    else:
        try:
            texts = load_and_chunk(file_pdf)
            retriver = create_embeddings_and_vectordb(texts)
        except Exception as e:
            st.error(f"An error occurred: {e}")
