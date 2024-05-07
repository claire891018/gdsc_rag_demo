FROM python:3.11.7-slim

EXPOSE 8501

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY . .

RUN pip install --upgrade pip \
    && pip install -r src/requirements.txt

# RUN pip install --no-cache-dir -r src/requirements.txt

# 設定 Streamlit 啟動指令，固定連接埠為 8501
CMD streamlit run src/home.py --server.port 8501