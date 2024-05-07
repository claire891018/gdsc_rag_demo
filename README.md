# RAG + Streamlit Demo 

## Description 
**RAG 概要**

檢索增強生成（Retrieval-Augmented Generation, RAG）是一種結合**搜尋檢索**和**生成能力**的自然語言處理架構。透過這個架構，**模型可以從外部知識庫搜尋相關信息**，然後**使用這些信息來生成回應或完成特定的NLP任務**。

**為什麼需要 RAG ?**

**大型語言模型（Large Language Models, LLMs）：通過投餵大量數據資料訓練**

- ㄧ定程度地理解人類使用語言
- 能夠與使用者互動
- 幫助解決問題

**大型語言模型的局限性：幻覺 Hallucination**

- 缺乏真正的理解能力 → 面對新知或異常情況時，無法正確回答解答（＝說屁話）

## RAG 技術說明

RAG 架構主要由兩個部分構成：retriever 檢索器和 generator 生成器。**檢索器**負責從外部知識庫**（你新給的資料）**中檢索相關的知識訊息。這些檢索到的知識將會被送到生成器進行處理。而**生成器會利用檢索到的知識來生成回應**。

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/06698bc9-025b-4282-a9f1-dbf294cb401b/13847ab4-adf1-41af-90df-6571eddb3da1/Untitled.png)

## Retriever

- type （[參考](https://python.langchain.com/docs/modules/data_connection/retrievers/)）// Utilized type - Vector store
- chain type 說明 (說明[參考](https://hackmd.io/@YungHuiHsu/BJ10qunzp))([操作參考](https://medium.com/@onkarmishra/using-langchain-for-question-answering-on-own-data-3af0a82789ed))

## streamlit

是開源 python 框架，不需要任何網頁的知識，只需要用 python 的語法（只能用 .py 建立）就能輕鬆架構出dashboard與app！

- **特點：**

1. **快速開發**：Streamlit的設計目的是快速從Python腳本轉換到互動式Web應用。
2. **易於使用**：使用簡單的Python語法，不需要前端經驗。
3. **高度互動性**：內置多種小組件（如滑動條、按鈕和選擇框），使用者可以輕鬆地與應用互動。
4. **實時更新**：當腳本中的代碼更改並保存時，Streamlit應用會自動更新，無需重新啟動。
5. **廣泛的可視化支持**：可以輕鬆集成Matplotlib、Plotly、Altair等流行的數據可視化庫。
    1. Demo：
    
    https://doc-hello.streamlit.app/?embed=true
    

- Streamlit 語法入門

安裝

```bash
pip install streamlit
```

本地 demo

```bash
streamlit run <你的檔案名稱.py>
```

https://cheat-sheet.streamlit.app/?embed=true

官方文件及Demo:

[Streamlit API cheat sheet - Streamlit Docs](https://docs.streamlit.io/develop/quick-reference/cheat-sheet)

- 關於部署

最快的方式是用github，建議用 github codespace 進行開發

[Deploy your app - Streamlit Docs](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app)
