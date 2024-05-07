# 套件導入
import streamlit as st
import os, tempfile
from pathlib import Path
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

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

import app_component as ac 
import utils as u

# 資料暫存資料夾
#temp_file = st.secrets["path"]
#TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
#LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')

def load_documents():
    loader = DirectoryLoader(file_pdf, glob='**/*.pdf')
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

# 初始化 Google Drive 服務
def init_google_drive_service():
    # 從 Streamlit 的 secrets 直接取得配置
    client_config = {
        "web": {
            "client_id": st.secrets["google_oauth"]["client_id"],
            "project_id": st.secrets["google_oauth"]["project_id"],
            "auth_uri": st.secrets["google_oauth"]["auth_uri"],
            "token_uri": st.secrets["google_oauth"]["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["google_oauth"]["auth_provider_x509_cert_url"],
            "client_secret": st.secrets["google_oauth"]["client_secret"]
        }
    }
    # 使用配置創建一個 Flow 實例
    flow = Flow.from_client_config(
        client_config,
        scopes=['https://www.googleapis.com/auth/drive.file']
    )
    # 執行身份驗證流程
    if 'credentials' not in st.session_state:
        # 需要手動開啟授權的 URL 並接收授權碼
        auth_url, _ = flow.authorization_url(prompt='consent')
        st.write("請去以下網址授權:", auth_url)
        code = st.text_input('輸入授權碼:')
        if code:
            flow.fetch_token(code=code)
            st.session_state.credentials = flow.credentials
    creds = Credentials(**st.session_state.credentials)
    return build('drive', 'v3', credentials=creds)

# 上傳文件到 Google Drive
def upload_file_to_drive(service, file_path, parent_id):
    file_metadata = {
        'name': os.path.basename(file_path),
        'parents': [parent_id]
    }
    media = MediaFileUpload(file_path, mimetype='application/pdf')
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    return file.get('id')

# 處理文件
def process_documents():
    if not google_api_key:
        st.warning("請上傳你的 api key！")
    else:
        try:
            drive_service = init_google_drive_service()
            folder_id = "test_temp"
            for uploaded_file in st.session_state.source_docs:
                # 將文件保存到臨時位置
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_file_path = temp_file.name
                # 上傳到 Google Drive
                file_id = upload_file_to_drive(drive_service, temp_file_path, folder_id)
                # 在此添加文件處理邏輯
                os.remove(temp_file_path)  # 刪除臨時文件
        except Exception as e:
            st.error(f"An error occurred: {e}")
