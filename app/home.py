import streamlit as st 
import app_utils as vutil 
import app_component as ac 


st.set_page_config(
    page_title="菜就多練",
    page_icon="https://api.dicebear.com/5.x/bottts-neutral/svg?seed=gptLAb"#,
    #menu_items={"About": "GPT Lab is a user-friendly app that allows anyone to interact with and create their own AI Assistants powered by OpenAI's GPT-3 language model. Our goal is to make AI accessible and easy to use for everyone, so you can focus on designing your Assistant without worrying about the underlying infrastructure.", "Get help": None, "Report a Bug": None}
)



ac.render_cta()

# copies 
home_title = "菜就多練"
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



