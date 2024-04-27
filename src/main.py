import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import LlamaCpp
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceHub
from htmlTemplates import css , bot_template , user_template

def get_pdf_text(upload_file):
    text = ""
    for pdf in upload_file:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text,chunk_size,chunk_overlap):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings =HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs = {'device': "cpu"},
        encode_kwargs = {'normalize_embeddings': True}
        )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore,model_path):
    if(st.session_state.endpoint == "HuggingFace"):
        
        llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2", 
            model_kwargs={"temperature":0.5, "max_length":512},
        )
    elif(st.session_state.endpoint == "Local LLM"):
        llm = LlamaCpp(
            steaming = True,
            # model_path ="/Users/zhangchenwei/Downloads/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            model_path =model_path,
            temperature = 0.7,
            top_p = 1,
            verbose = True,
            n_ctx = 40960,
        )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    if(st.session_state.endpoint == "HuggingFace"):
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                index = message.content.find("Helpful Answer:")
                content = message.content[index + len("Helpful Answer:"):].strip()
                st.write(bot_template.replace(
                    "{{MSG}}", content), unsafe_allow_html=True)
                
    elif(st.session_state.endpoint == "Local LLM"):
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
        

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
                
    # if user_question is not None and user_question !="":
    #     st.session_state.chat_history.append(HumanMessage(user_question))
    #     with st.chat_message("Human"):
    #         st.markdown(user_question)
    # response = st.session_state.conversation({"question": user_question})
    # st.session_state.chat_history = response['chat_history']
    # st.write(response)
    # for message in st.session_state.chat_history:
    #     if isinstance(message, HumanMessage):
    #         with st.chat_message("Human"):
    #             st.markdown(message.content)
    #     else:        
    #         with st.chat_message("AI"):
    #             st.markdown(message.content)

def clear_history():
    st.session_state.chat_history = []

def main():
    load_dotenv()
    st.set_page_config(page_title="streambot",page_icon="🤖")
    st.write(css, unsafe_allow_html=True)
    st.title("RAG bot 🤖")
    if "conversation" not in st.session_state:
        st.session_state.conversation = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ""
    if "model_path" not in st.session_state:
        st.session_state.model_path = ""
    if "endpoint" not in st.session_state:
        st.session_state.endpoint = "HuggingFace"
    st.header("chat with PDF")
    user_question = st.chat_input("enter your question")
    if user_question:
        handle_userinput(user_question)
        
    with st.sidebar :
        st.session_state.endpoint = st.selectbox("LLM endpoint", ["HuggingFace", "Local LLM"])
        if st.session_state.endpoint == "HuggingFace":
            api_key = st.text_input("HuggingFace API Key", type="password",value=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
            if api_key=="None" or api_key == "" or api_key is None:
                st.warning("Check API KEY.")
            else:
                with open('.env', 'w') as f:
                    f.write(f'HUGGINGFACEHUB_API_TOKEN={api_key}\n')
                    st.success("API KEY Exist.")
                    st.empty() 
        elif st.session_state.endpoint == "Local LLM":
            st.session_state.model_path = st.text_input("Path of LLM model")
            if st.session_state.model_path == "" or st.session_state.model_path is None:
                st.warning("Require model path.")
            else:
                st.empty()
 

        upload_file = st.file_uploader("upload file" , type = ["pdf", "docx", "txt"],accept_multiple_files=True)    
        chunk_size = st.number_input("chunk size", min_value = 100, max_value=2048, value=200, on_change=clear_history)
        chunk_overlap = st.number_input("chunk overlap", min_value = 1, max_value=200, value=50, on_change=clear_history)
        if st.button("upload data", on_click=clear_history):
            with st.spinner("uploading..."):  
            # get pdf text
                raw_text = get_pdf_text(upload_file)

            # get text chunks
                text_chunks = get_text_chunks(raw_text, chunk_size, chunk_overlap)
    
            # create vector store
                vectorstore = get_vectorstore(text_chunks)
            #create conversation
                
                st.session_state.conversation = get_conversation_chain(vectorstore,st.session_state.model_path)           
    
if __name__ == "__main__":
    main()


# get response from LLM
# def get_response(user_query, chat_history):
#     template = """
#     You are a helpful assistant. Answer the following questions considering the history of the conversation:

#     Chat history: {chat_history}

#     User question: {user_question}
#     """

#     prompt = ChatPromptTemplate.from_template(template)

#     #llm = ChatOpenAI()
#     llm = LlamaCpp(
#         steaming = True,
#         model_path ="/Users/zhangchenwei/Downloads/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
#         temperature = 0.7,
#         top_p = 1,
#         verbose = True,
#         n_ctx = 4096,
#     )

#     chain = prompt | llm | StrOutputParser()

#     return chain.stream({
#         "chat_history": chat_history,
#         "user_question": user_query,
#     })

    
# for message in st.session_state.chat_history:
#     if isinstance(message, HumanMessage):
#         with st.chat_message("Human"):
#             st.markdown(message.content)
#     else:
#         with st.chat_message("AI"):
#             st.markdown(message.content)

# user_query = st.chat_input("your message")
# if user_query is not None and user_query !="":
#     st.session_state.chat_history.append(HumanMessage(user_query))
    
#     with st.chat_message("Human"):
#         st.markdown(user_query)

#     with st.chat_message("AI"):
#         ai_response = st.write_stream(get_response(user_query, st.session_state.chat_history))

#     st.session_state.chat_history.append(AIMessage(ai_response))