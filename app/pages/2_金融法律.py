import streamlit as st 
import app_component as ac 
import app_utils as au 
import requests

st.set_page_config(
    page_title="金融法律",
    page_icon="https://api.dicebear.com/5.x/bottts-neutral/svg?seed=gptLAb"#,
    #menu_items={"About": "GPT Lab is a user-friendly app that allows anyone to interact with and create their own AI Assistants powered by OpenAI's GPT-3 language model. Our goal is to make AI accessible and easy to use for everyone, so you can focus on designing your Assistant without worrying about the underlying infrastructure.", "Get help": None, "Report a Bug": None}
)

st.markdown(
    "<style>#MainMenu{visibility:hidden;}</style>",
    unsafe_allow_html=True
)

    # input_text = st.text_input("Your question:")
    # if st.button("Ask"):
    #     response = call_finance_knowledge_bot(input_text)
    #     st.write("Bot:", response)

# Call RAG API
def call_rag_api(prompt):
    data = {"prompt": prompt}
    headers = {"Authorization": f"Bearer {st.secrets['API_TOKEN']}"}
    response = requests.post("https://your-api-url.com/endpoint", json=data, headers=headers)
    return response.json()

def generate_response():
    response = call_rag_api(prompt)
    return response.get("response_text", "Sorry, I couldn't process that.")

st.title("金融知識")
#st.markdown("---")
ac.robo_avatar_component()
st.markdown("  \n")

# Display messages
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "菜鳥你好，歡迎詢問有關金融法律ㄉ問題！"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
prompt = st.chat_input("在此輸入問題...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("思考中..."):
        response_text = generate_response()
        st.session_state.messages.append({"role": "assistant", "content": response_text})

# Clear chat history
def clear_chat():
    st.session_state.messages = []

st.sidebar.button("Clear chat", on_click=clear_chat)