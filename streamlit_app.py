%%writefile chatbot.py

import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from langchain.prompts import ChatPromptTemplate
import tempfile
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_tool_calling_agent, AgentExecutor

# .env 파일 로드
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def search_web():
    # TavilySearchResults 클래스의 인스턴스를 생성
    web_search = TavilySearchResults(k=6)
    return web_search

def load_pdf_files(uploaded_files):
    all_documents = []
    for uploaded_file in uploaded_files:
        # 업로드된 파일을 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # PyPDFLoader로 임시 파일 로드
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()

        all_documents.extend(documents)

    # 텍스트 분할기 설정
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(all_documents)

    # FAISS 인덱스 설정 및 생성
    vector = FAISS.from_documents(split_docs, OpenAIEmbeddings())
    retriever = vector.as_retriever()

    # 도구 정의
    retriever_tool = create_retriever_tool(
        retriever,
        name="pdf_search",
        description="Use this tool to search information from the pdf document"
    )
    return retriever_tool

# 에이전트와 대화하는 함수
def chat_with_agent(user_input, agent_executor):
    result = agent_executor({"input": user_input})
    response = result['output']  # 명시적으로 출력 키를 처리
    return response

# 세션 기록 가져오는 함수
def get_session_history(session_ids):
    if session_ids not in st.session_state.session_history:
        st.session_state.session_history[session_ids] = ChatMessageHistory()
    return st.session_state.session_history[session_ids]

# 대화 내용 출력하는 함수
def print_messages():
    for msg in st.session_state["messages"]:
        st.chat_message(msg['role']).write(msg['content'])

# Streamlit 메인 코드
def main():
    # 페이지 설정
    st.set_page_config(page_title="AI 비서", layout="wide", page_icon="🤖")

    st.image('/content/KB_Sec.png', width=1200)
    st.markdown('---')
    st.title("안녕하세요! RAG를 활용한 'KB증권 AI 비서' 입니다")  # 시작 타이틀

    # 세션 초기화
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "session_history" not in st.session_state:
        st.session_state["session_history"] = {}

    with st.sidebar:
        st.session_state["OPENAI_API"] = st.text_input(label="OPENAI API 키", placeholder="Enter Your API Key", value="", type="password")
        st.session_state["TAVILY_API"] = st.text_input(label="TAVILY_API 키", placeholder="Enter Your API Key", value="", type="password")
        st.markdown('---')
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, key="pdf_uploader")

    # OpenAI API 키가 입력되었는지 확인
    if st.session_state["OPENAI_API"] and st.session_state["TAVILY_API"]:
        os.environ['OPENAI_API_KEY'] = st.session_state["OPENAI_API"]
        os.environ['TAVILY_API_KEY'] = st.session_state["TAVILY_API"]
        if pdf_docs:
            pdf_search = load_pdf_files(pdf_docs)
            web_search = search_web()
            tools = [pdf_search, web_search]

            # LLM 설정
            llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

            # 프롬프트 설정
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system",
                     "Be sure to answer in Korean. You are a helpful assistant. "
                    "Make sure to use the `pdf_search` tool for searching information from the pdf document."
                    "Please always include emojis in your responses with a friendly tone."
                    "If you can't find the information from the PDF document, use the `web_search` tool for searching information from the web."
                    "Your name is `KB증권 AI 비서`. Please introduce yourself at the beginning of the conversation."),
                    ("placeholder", "{chat_history}"),
                    ("human", "{input} \n\n Be sure to include emoji in your responses."),
                    ("placeholder", "{agent_scratchpad}"),
                ]
            )

            # 에이전트 생성 (initialize_agent 대신 create_tool_calling_agent 사용)
            agent = create_tool_calling_agent(llm, tools, prompt)

            # AgentExecutor 정의
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

            # 사용자 입력 처리
            user_input = st.chat_input('질문이 무엇인가요?')

            if user_input:
                session_id = "default_session"
                session_history = get_session_history(session_id)

                if session_history.messages:
                    previous_messages = [{"role": msg['role'], "content": msg['content']} for msg in session_history.messages]
                    response = chat_with_agent(user_input + "\n\nPrevious Messages: " + str(previous_messages), agent_executor)
                else:
                    response = chat_with_agent(user_input, agent_executor)

                # 메시지를 세션에 추가
                st.session_state["messages"].append({"role": "user", "content": user_input})
                st.session_state["messages"].append({"role": "assistant", "content": response})

                # 세션 기록에 메시지를 추가
                session_history.add_message({"role": "user", "content": user_input})
                session_history.add_message({"role": "assistant", "content": response})

            # 대화 내용 출력
            print_messages()

    else:
        st.warning("OpenAI API 키를 입력하세요")

if __name__ == "__main__":
    main()
