import streamlit as st
import component as ac 
import PyPDF2
import os
import tempfile
from langchain.document_loaders import PyMuPDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from sentence_transformers import SentenceTransformer
import pysqlite3
import sys

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    # read documents
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())

    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # create embeddings and store in vectordb
    model_name = "aspire/acge_text_embedding"
    model_kwargs = {'device': 'cpu'}
    embeddings = HuggingFaceEmbeddings(model_name=model_name,
                                      model_kwargs=model_kwargs)
    vectordb = Chroma.from_documents(splits, embeddings)

    # define retriever
    retriever = vectordb.as_retriever(
        search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10}
    )
    return retriever  
    
#### Page ####
st.set_page_config(
    page_title="智慧機器人",
    page_icon="https://api.dicebear.com/8.x/bottts/svg?seed=Felix"
)
    
home_title = "0508 GDSC 智慧機器人"
st.markdown(f"""# {home_title} <span style=color:#2E9BF5><font size=5>Beta</font></span>""",unsafe_allow_html=True)
ac.robo_avatar_component()
mode = st.sidebar.radio("LLM type：", ('上傳你的 Google API Key',))
if mode == '上傳你的 Google API Key':
    google_api_key = st.sidebar.text_input('Google API Key:', type='password')

# File upload
uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF files", type=["pdf"], accept_multiple_files=True
)
if not uploaded_files:
    st.info("請上傳資料")
    st.stop()

retriever = configure_retriever(uploaded_files)
llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key)
template = """
    資料：{context}
    你能協助閱讀資料，並且總是能夠立即、準確地回答任何要求。
    請用繁體中文(traditonal chinese)回答下列問題。不知道就回答「資訊未提及：{question}」，並確保答案內容相關且簡潔。
    問題：{question}
    回答：
"""
prompt = PromptTemplate.from_template(template=template)
qa_Google = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # can be adjusted
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
    #verbose=True 控制輸出詳細程度
)
old = """
msgs = StreamlitChatMessageHistory()

if len(msgs.messages) == 0 or st.sidebar.button("新對話"):
    msgs.clear()
    msgs.add_ai_message("需要什麼協助～")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(query)

    with st.chat_message("assistant"):
        response = qa_Google.invoke({"query": query})
        ## print answer
        answer = response["result"]
        st.write(answer)
"""

# Display messages
if "msgs" not in st.session_state:
    st.session_state.msgs = [{"role": "assistant", "content": "你好，歡迎詢問有關ㄉ問題！"}]

for message in st.session_state.msgs:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if query := st.chat_input("請輸入問題..."):
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.msgs.append({"role": "user", "content": query})
    with st.spinner("思考中..."):
        response = qa_Google.invoke({"query": query})
        ## print answer
        answer = response["result"]
        st.session_state.msgs.append({"role": "assistant", "content": f'{answer}'})
    with st.chat_message("assistant"):
        st.markdown(answer)

# Clear chat history
def clear_chat():
    st.session_state.msgs = []

st.sidebar.button("Clear chat", on_click=clear_chat)




