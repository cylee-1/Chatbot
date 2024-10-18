__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain

st.set_page_config(page_title="LoL Academy ChatBot")
st.title('LoL Academy ChatBot')

#환경변수 사용
st.secrets["OPENAI_API_KEY"]
st.secrets["ENDPOINT"]

#따로 학습시킬 텍스트 파일 로드
documents = UnstructuredFileLoader("lol.txt")

#캐시로 저장
cache_dir = LocalFileStore("./.cache_lol/")

#텍스트 분할
splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n",
    chunk_size=300,
    chunk_overlap=50
)

docs = documents.load_and_split(text_splitter=splitter)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
vectorstore = Chroma.from_documents(docs, cached_embeddings, persist_directory="./.verctorDB_lol")

llm=ChatOpenAI(
        temperature=0.5,
        model_name='gpt-3.5-turbo'
        )

#질문 보내고 받기
def generate_response(input_text):
    chain = load_qa_chain(llm, chain_type="stuff", verbose=True)
    matching_docs = vectorstore.similarity_search(input_text)
    answer = chain.run(input_documents=matching_docs, question=input_text)
    st.info(answer)

#페이지 form
with st.form('Qusetion'):
    text = st.text_area('질문 입력 : ')
    submitted = st.form_submit_button('보내기')
    generate_response(text)