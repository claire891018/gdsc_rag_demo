import os, tempfile
from pathlib import Path

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

import streamlit as st
import app_component as ac 

# 定義全局變數
TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')

if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)

def load_documents():
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf')
    documents = loader.load()
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    texts = text_splitter.split_documents(documents)
    return texts

def google_llm(google_api_key):
    llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key)
    return llm

def embeddings_on_local_vectordb(texts):
    model_name = "aspire/acge_text_embedding"
    model_kwargs = {'device': 'cpu'}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    vectordb = Chroma.from_documents(texts, embedding=embeddings, persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix())
    vectordb.persist()
    retriever = vectordb.as_retriever(search_kwargs={'k': 7})
    return retriever

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
    你能協助閱讀資料，並且總是能夠立即、準確地回答任何要求。
    請用繁體中文(traditonal chinese)回答下列問題。不知道就回答「不知道」，並確保答案內容相關且簡潔：
    {query}
    """
    input_prompt = PromptTemplate(input_variables=["query"], template=init_Prompt)
    return LLMChain(prompt=input_prompt, llm=llm)

def input_fields():
    st.session_state.source_docs = st.file_uploader(label="上傳文件！", type="pdf", accept_multiple_files=True)

def process_documents():
    if not google_api_key:
        st.warning(f"請上傳你的 api key！")
    else:
        try:
            for source_doc in st.session_state.source_docs:
                with tempfile.TemporaryFile(delete=False, dir=TMP_DIR.as_posix(), suffix='.pdf') as tmp_file:
                    tmp_file.write(source_doc.read())
                documents = load_documents()
                for _file in TMP_DIR.iterdir():
                    temp_file = TMP_DIR.joinpath(_file)
                    temp_file.unlink()
                texts = split_documents(documents)
                st.session_state.retriever = embeddings_on_local_vectordb(texts)
        except Exception as e:
            st.error(f"An error occurred: {e}")

def boot():
    input_fields()
    st.button("上傳文件", on_click=process_documents())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])
    if query := st.chat_input("在此輸入問題..."):
        st.chat_message("human").write(query)
        if "retriever" in st.session_state:
            response = query_llm(st.session_state.retriever, query)
        else:
            response = query_llm_direct(query)
        st.chat_message("ai").write(response)

#with st.spinner("思考中..."):

if __name__ == '__main__':
    st.set_page_config(
    page_title="智慧機器人",
    page_icon="https://api.dicebear.com/8.x/bottts/svg?seed=Felix"
    )
    st.title("0508 GDSC")
    home_title = "智慧機器人"
    st.markdown(f"""# {home_title} <span style=color:#2E9BF5><font size=5>Beta</font></span>""",unsafe_allow_html=True)
    ac.robo_avatar_component()
    mode = st.sidebar.radio("LLM type：", ('上傳你的 Google API Key',))
    if mode == '上傳你的 Google API Key':
        google_api_key = st.sidebar.text_input('Google API Key:', type='password')
    boot()



