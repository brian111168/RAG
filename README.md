# RAG


## Introduction

可以上傳pdf檔案，並且和LLM進行問答，使用的語言模型是Mistral-7B-Instruct-v0.2

## Installation

To install and run , follow these steps:

1. Clone the Repository:`git clone https://github.com/brian111168/RAG.git`
2. `cd RAG`
3. `conda create -n RAG python=3.11`
4. `conda activate RAG`
5. 在python專案中建立一個.env檔案
6. 開啟Vscode，選取剛剛create的環境，RAG（ctrl+shft+P）
7. 開啟VScode終端機，安装環境:`pip install -r requirements.txt`，目前會遇到llama-cpp-python無法下載，先將requirements.txt 中的 llama_cpp_python==0.2.55刪除

## Run the Application

Execute the following command to start the Streamlit application:

1. `streamlit run main.py`
This command will launch the Streamlit app, and you can access it via your web browser. main.py更換為檔案的路徑如：`streamlit run /Users/zhangchenwei/Desktop/RAG/src/main.py`

2. Select LLM Endpoint: Choose between "HuggingFace" and "Local LLM" as the endpoint for the language model.目前使用的 Endpoint 是 "HuggingFace"
3. create Access Tokens:`https://huggingface.co/settings/tokens`
4. 將 Access Tokens 貼至 HuggingFace API Key 欄位
5. 上傳PDF檔案至 upload file 區域，可進行 RAG 問答
