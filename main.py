import streamlit as st
import app_component as ac 

import utils as u

def input_fields():
    st.session_state.source_docs = st.file_uploader(label="上傳文件！", type="pdf", accept_multiple_files=True)

def process_documents():
    if not google_api_key:
        st.warning("請上傳你的 api key！")
    else:
        try:
            for source_doc in st.session_state.source_docs:
                texts = u.load_and_chunk(source_doc)
                st.session_state.retriever = u.create_embeddings_and_vectordb(texts)
        except Exception as e:
            st.error(f"An error occurred: {e}")

def boot():
    st.button("上傳文件", on_click=process_documents())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])
    if query := st.chat_input("在此輸入問題..."):
        st.chat_message("human").write(query)
        if "retriever" in st.session_state:
            response = u.query_llm(st.session_state.retriever, query)
        else:
            response = u.query_llm_direct(query)
        st.chat_message("ai").write(response)

#with st.spinner("思考中..."):

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
    input_fields()
    #file_pdf = st.file_uploader(label="上傳文件！", type="pdf", accept_multiple_files=True)
    boot()



