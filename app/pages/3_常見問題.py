import streamlit as st 
import app_component as au

st.set_page_config(
    page_title="常見問題",
    page_icon="https://api.dicebear.com/5.x/bottts-neutral/svg?seed=gptLAb"#,
    #menu_items={"About": "GPT Lab is a user-friendly app that allows anyone to interact with and create their own AI Assistants powered by OpenAI's GPT language model. Our goal is to make AI accessible and easy to use for everyone, so you can focus on designing your Assistant without worrying about the underlying infrastructure.", "Get help": None, "Report a Bug": None}
)


st.markdown(
    "<style>#MainMenu{visibility:hidden;}</style>",
    unsafe_allow_html=True
)

au.render_cta()

st.title("常見問題")

#st.markdown("---")
au.robo_avatar_component()

st.markdown("#### 關於我們")
with st.expander("Bulldog.app 是什麼?", expanded=False):
    st.markdown("""
    回覆
    """)

with st.expander("採用的 AI 模型？"):
    st.markdown("""
    「Bulldog.app」採用 Gemini pro
    """)

st.markdown("#### 常見問題")

with st.expander("怎麼操作", expanded=False):
    st.markdown("""
    回覆
    """)
