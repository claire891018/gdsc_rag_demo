import streamlit as st
import component as ac 
import PyPDF2
import os

import streamlit as st
import tempfile
from langchain.document_loaders import PyMuPDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

@st.cache(allow_output_mutation=True)

def generator(uploaded_file, google_api_key, query):
    # Load document if file is uploaded
    if uploaded_file is not None:
        try:
            temp_dir = tempfile.mkdtemp()
            path = os.path.join(temp_dir, uploaded_file.name)
            with open(path, "wb") as f:
                f.write(uploaded_file.getvalue())
            #temp_file = "./temp.pdf"
            #with open(temp_file, "wb") as file:
            #   file.write(uploaded_file.getvalue())
            #   file_name = uploaded_file.name
            loader = PyPDFLoader(path)
            #loader = PyMuPDFLoader(uploaded_file.name)
            documents = loader.load()
        
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
            texts = text_splitter.create_documents(documents)
            all_splits = text_splitter.split_documents(documents)
            
            # Select embeddings
            model_name = "aspire/acge_text_embedding"
            model_kwargs = {'device': 'cpu'}
            embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
            # Create a vectorstore from documents
            db = Chroma.from_documents(all_splits, embeddings, persist_directory="db")
            # Create retriever interface
            retriever = db.as_retriever()
            
            llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key)
            template = """
            資料：{context}
            你能協助閱讀資料，並且總是能夠立即、準確地回答任何要求。
            請用繁體中文(traditonal chinese)回答下列問題。不知道就回答「資訊未提及：{query}」，並確保答案內容相關且簡潔。
            問題：{query}
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
            result = qa_Google.invoke({"query": query})
            response = result["result"]
            return response

        except Exception as e:
            st.error(f"資料解析失敗，原因：{e}")
            return None
    
def boot(): 
    # File upload
    uploaded_file = st.file_uploader('請上傳資料', type='pdf')

    # Query text
    if "messages" not in st.session_state or st.sidebar.button("清除歷史資料"):
        st.session_state["messages"] = [{"role": "assistant", "content": "請根據資料提問"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if query := st.chat_input(placeholder="詢問跟資料相關的問題！"):
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").write(query)

        if not google_api_key:
            st.warning("請上傳你的 api key！")
            st.stop()

        with st.chat_message("assistant"):
            response = generator(uploaded_file, google_api_key, query)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)

if __name__ == '__main__':
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
    boot()



