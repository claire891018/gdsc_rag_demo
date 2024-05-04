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

## 

## 

## Contributions
Contributions are welcomed. Simply open up an issue and create a pull request. If you are introducing new features, please provide a detailed description of the specific use case you are addressing and set up instructions to test. 

Aside: I am new to open source, work full-time,  and have young kids, please bear with me if I don't get back to you right away. 

## License
This project is licensed under the terms of the MIT License. See [LICENSE](LICENSE) for more details.
