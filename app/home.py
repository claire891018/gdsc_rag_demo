import streamlit as st 
import app_component as ac 
import app.main as re 


st.set_page_config(
    page_title="智慧機器人",
    page_icon="https://api.dicebear.com/8.x/bottts/svg?seed=gptLAb"
)


ac.render_cta()

# copies 
home_title = "智慧機器人"
home_introduction = "請在左側欄位選擇你想詢問的主題"
home_privacy = "安全"

st.markdown(
    "<style>#MainMenu{visibility:hidden;}</style>",
    unsafe_allow_html=True
)

st.markdown(f"""# {home_title} <span style=color:#2E9BF5><font size=5>Beta</font></span>""",unsafe_allow_html=True)

st.markdown("""\n""")
st.markdown("#### 菜鳥你好")
st.write(home_introduction)

#st.markdown("---")
ac.robo_avatar_component()

st.markdown("#### 隱私權說明")
st.write(home_privacy)

st.markdown("""\n""")
st.markdown("""\n""")

st.markdown("#### Get Started")

# Display messages
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "想問什麼問題？"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
prompt = st.chat_input("在此輸入問題...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("思考中..."):
        response_text = re.generate_response()
        st.session_state.messages.append({"role": "assistant", "content": response_text})

# Clear chat history
def clear_chat():
    st.session_state.messages = []

st.sidebar.button("Clear chat", on_click=clear_chat)

st.sidebar.header("Google LLM Settings")
api_key = st.sidebar.text_input("Enter your Google LLM API Key:")

# 確保 API 金鑰不為空
if api_key:
    # 初始化 Google LLM
    llm_instance = re.Google_llm(api_key=api_key)

    # 在這裡使用 llm_instance 進行後續操作
    st.success("Google LLM Initialized Successfully!")
else:
    st.warning("Please enter your Google LLM API Key.")



