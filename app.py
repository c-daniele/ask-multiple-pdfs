import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
#from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_openai import OpenAIEmbeddings
#from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
#from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_community.llms import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_community.llms import HuggingFaceHub
from tempfile import NamedTemporaryFile
#import boto3
from libs import *

def get_pdf_docs(pdf_docs):
    docs = []
    for pdf in pdf_docs:
        with NamedTemporaryFile(dir='.', suffix='.csv') as f:
            f.write(pdf.getbuffer())
            loader = PyPDFLoader(f.name)
            docs.extend(loader.load_and_split())
    return docs


def get_text_chunks(docs):
    #text_splitter = CharacterTextSplitter(
    #    separator="\n",
    #    chunk_size=1000,
    #    chunk_overlap=200,
    #    length_function=len
    #)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    return chunks


def get_vectorstore(text_chunks):
    embeddings_model = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    #vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorstore = Chroma.from_documents(documents=text_chunks, embedding=embeddings_model)
    return vectorstore


def get_conversation_chain(vectorstore):
    #myLLM = create_llm("amazon.titan-text-express-v1")
    print(f"SELECTED MODEL: {st.session_state.selected_llm['value']}")
    
    myLLM = create_llm(model_name=st.session_state.selected_llm['value'])

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=myLLM,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation.invoke({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def formatLLMSelectBox(val):
    return val['value']

def main():
    load_dotenv()
    llms = getAvailableLLMs()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:

        st.subheader("Select the LLM")
        llm_select = st.selectbox('Available LLM', llms, index=None, key='selected_llm', format_func= formatLLMSelectBox)

        st.subheader("Your documents")
        pdf_files = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process", disabled = (st.session_state['selected_llm'] == None)):
            with st.spinner("Processing"):
                # get pdf text
                docs = get_pdf_docs(pdf_files)

                # get the text chunks
                text_chunks = get_text_chunks(docs)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()
