import streamlit as st 
import app_component as au

st.set_page_config(
    page_title="Terms",
    page_icon="https://api.dicebear.com/5.x/bottts-neutral/svg?seed=gptLAb"#,
    #menu_items={"About": "GPT Lab is a user-friendly app that allows anyone to interact with and create their own AI Assistants powered by OpenAI's GPT language model. Our goal is to make AI accessible and easy to use for everyone, so you can focus on designing your Assistant without worrying about the underlying infrastructure.", "Get help": None, "Report a Bug": None}
)

st.markdown(
    "<style>#MainMenu{visibility:hidden;}</style>",
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown('''
    - [Terms of Use](#terms-of-use)
    - [Privacy Policy](#privacy-policy)
    ---
    ''', unsafe_allow_html=True)

au.render_cta()    

st.title("使用條款")
st.write("日期：")
st.write("### Bulldog.app")

st.info("""
平台專為國泰新人打造。
請同意並遵守以下使用條款！
""")

st.markdown("""
##### 1. 第一條  \n 
ＯＯＯＯ
            
##### 2. 第二條 \n
ＯＯＯＯ
            
""")
