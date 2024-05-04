# RAG Lab 

## Description 
GPT Lab is a user-friendly [Streamlit](https://streamlit.io) app that lets users interact with and create their own AI Assistants powered by OpenAI's GPT language model. With GPT Lab, users can chat with pre-built AI Assistants or create their own by specifying a prompt and OpenAI model parameters. Our goal is to make AI accessible and easy to use for everyone, so users can focus on designing their Assistants without worrying about the underlying infrastructure.

[RAG working demo](https://colab.research.google.com/drive/1taBL5333ZmBXMoFuEoUGs43QUDpgj433?usp=sharing)

GPT Lab is also featured in the [Streamlit App Gallery](https://streamlit.io/gallery) among other impressive Streamlit apps.

For more insight into the development process and lessons learned while building GPT Lab, check out this [blog post](https://blog.streamlit.io/building-gpt-lab-with-streamlit/) on the official Streamlit blog.

This README will cover:
- Data models
- Accessing the app
- Running the app Locally
- Contributions
- License

## Data models

```
Users Collection
   |
   | - id: (Firestore auto-ID)
   | - user_hash: string (one-way hash value of OpenAI API key)
   | - created_date: datetime
   | - last_modified_date: datetime
   | - sessions_started: number
   | - sessions_ended: number
   | - bots_created: number
```

## Accessing the app 
You can access the app on the Streamlit Cloud community at [gptlab.streamlit.app](https://gptlab.streamlit.app/). 

To use the app, you will need an OpenAI API key. Don't have one yet? Create one on [the OpenAI website](https://platform.openai.com/account/api-keys). Once you have your API key, enter it into the app when prompted. 

For optimal chatting experience, we recommend using a pay-as-you-go API key. (Free trial API keys are limited to 3 requests a minute, not enough to chat with assistants.) You will need to enter your billing information [here](https://platform.openai.com/account/billing/overview). You can learn more about OpenAI API rate limits [here](https://platform.openai.com/docs/guides/rate-limits/overview).

## Running the app locally 

To run the app locally, you will need to: 

1. Set up your own [Google Firestore](https://firebase.google.com/docs/firestore) database. 
    - GPT Lab uses four main collections: `users`, `user_hash`, `bots`, and `sessions`.
    - You will need to manually set up a `users` collection before you can run the app locally. (All other collections will be set up by the app). 
2. Clone this repository
3. Create a .streamlit/secrets.toml file containing the following:
```
[firestore]
db-key = "YOUR GOOGLE SERVICE ACCOUNT TOML"

[util]
global_salt = "OPTIONAL GLOBAL SALT"
```

- You will need to generate a service account JSON and convert that JSON file to TOML. Follow the instructions [here](https://blog.streamlit.io/streamlit-firestore-continued/). 

4. In your terminal, set up your local environment: 
    - Set up a Python virtual environment (using `venv`, `conda`, `virtualenv`, or any other tool you prefer)
    - Install the required Python dependencies (`pip install -r app/requirements.txt`)
    - Run `streamlit run app/home.py`

## Contributions
Contributions are welcomed. Simply open up an issue and create a pull request. If you are introducing new features, please provide a detailed description of the specific use case you are addressing and set up instructions to test. 

Aside: I am new to open source, work full-time,  and have young kids, please bear with me if I don't get back to you right away. 

## License
This project is licensed under the terms of the MIT License. See [LICENSE](LICENSE) for more details.


# Bulldog.app

## Description 

Bulldog.app 是一個基於 Streamlit 建構的聊天機器人應用，旨在提供金融知識和金融法律的相關資訊。 該應用程式包含四個主要部分：首頁、金融知識聊天機器人、金融法律聊天機器人和常見問題（FAQ）。

這個 README 會包含:
- Fuctions
- Code structure
- Accessing the app
- Running the app Locally
- Contributions
- License

## Fuctions

- **首頁**: 引導使用者進入選擇主題。
- **金融知識**: 使用者可以透過聊天機器人形式查詢金融相關知識。
- **金融法律**: 使用者可以透過聊天機器人形式查詢金融相關法律。
- **常見問題**: 提供金融常見問題的答案。

## Accessing the app


## Description 
歡迎對專案做出貢獻！ 如果您有改進意見或功能增加的建議，請遵循以下步驟：

## License
本項目採用 MIT 許可證。 有關詳細信息，請查看 LICENSE 文件。

## Contact us
如果您有任何問題，請將問題傳於此「google form link」。

