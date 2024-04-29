from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
import streamlit as st

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたはAIチャットbotです。"),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="human_message")
    ]
)
LLM = ChatBedrock(model_id="anthropic.claude-v2:1", region_name="us-east-1", model_kwargs={"max_tokens": 4000})
chain = prompt | LLM

# 初回はsession領域を作成
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 2回目以降はsessionを元に全量再描画を行う
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 入力された場合
if user_prompt := st.chat_input():
    with st.chat_message("user"):
        st.write(user_prompt)
    with st.chat_message("assistant"):
        response = st.write_stream(chain.stream({"messages": st.session_state.messages, "human_message": [HumanMessage(content=user_prompt)]}))
    
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    st.session_state.messages.append({"role": "assistant", "content": response})

