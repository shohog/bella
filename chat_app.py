import os
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from pymongo import MongoClient
from uuid import uuid4

mongo_client = MongoClient(st.secrets["MONGODB_URI"])
db = mongo_client["bella"]
conversations_collection = db["conversations"]

llm = None

def format_name(name):
    return name.replace('_', ' ').replace('.txt', '').title()

def get_llm_instance(api_key):
    global llm
    if llm is None:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.5,
            max_output_tokens=8192,
            stream=True,
            google_api_key=api_key,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            },
        )
    return llm

def get_response(user_query, conversation_history, api_key, system_prompt, chapter_content):
    prompt_template = """
    {system_prompt}

    Chapter Content:
    {chapter_content}

    Chat History:
    {conversation_history}

    Student Question:
    {user_query}
    """
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    llm = get_llm_instance(api_key)
    chain = prompt | llm | StrOutputParser()
    return chain.stream({
        "system_prompt": system_prompt,
        "conversation_history": conversation_history,
        "chapter_content": chapter_content,
        "user_query": user_query,
    })

def save_to_mongodb(user_id, conversation, class_name, subject_name, chapter_name):
    conversations_collection.update_one(
        {"user_id": user_id},
        {
            "$set": {
                "user_id": user_id,
                "class": class_name,
                "subject": subject_name,
                "chapter": chapter_name,
            },
            "$push": {
                "conversation": {"$each": conversation}
            },
        },
        upsert=True
    )

st.set_page_config(page_title="Bella", page_icon=":robot:")

st.markdown("""
    <style>
        .centered-title {
            text-align: center;
            font-size: 2.5rem !important;
            font-weight: bold;
            margin-bottom: 2rem;
        }
        .stSelectbox {
            width: 50%;
            max-width: 200px;
            margin: 0 auto;
        }
        div[data-testid="column"] {
            text-align: center;
        }
        .stSelectbox label {
            font-weight: bold !important;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="centered-title">বেলা::Bella</p>', unsafe_allow_html=True)

api_key = st.secrets["GEMINI_API_KEY"]
base_dir = "./books/nine/"

if "previous_selection" not in st.session_state:
    st.session_state.previous_selection = None

col1, col2, col3 = st.columns(3)

with col1:
    selected_class = st.selectbox("Class", os.listdir(base_dir))
with col2:
    selected_subject = st.selectbox("Subject", 
        os.listdir(os.path.join(base_dir, selected_class)))
with col3:
    chapters = os.listdir(os.path.join(base_dir, selected_class, selected_subject))
    chapters_no_ext = [os.path.splitext(chapter)[0] for chapter in chapters]
    selected_chapter = st.selectbox("Chapter", chapters_no_ext)
    selected_chapter = next(ch for ch in chapters if ch.startswith(selected_chapter))

formatted_class = format_name(selected_class)
formatted_subject = format_name(selected_subject)
formatted_chapter = format_name(selected_chapter)

current_selection = (selected_class, selected_subject, selected_chapter)

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid4())

if "messages" not in st.session_state or st.session_state.previous_selection != current_selection:
    # greeting = f"Hey! I'm Bella.  Ready to explore chapter **{formatted_chapter}** in **{formatted_subject}** for class **{formatted_class}**?"
    
    greeting = f"হ্যালো! আমি বেলা ,তোমার নাম কী? লেখা পড়ার কী অবস্থা! চলো কিছুক্ষণ **{formatted_subject}** এর **{formatted_chapter}**পড়ি!" 

    st.session_state.messages = [AIMessage(content=greeting)]
    st.session_state.previous_selection = current_selection

user_id = st.session_state.user_id

chapter_path = os.path.join(base_dir, selected_class, selected_subject, selected_chapter)
with open(chapter_path, "r", encoding="utf-8") as file:
    chapter_content = file.read()

with open("bella_system.md", "r") as file:
    system_prompt = file.read().format(
        class_name=formatted_class,
        subject=formatted_subject,
        chapter=formatted_chapter,
        content=chapter_content
    )

for message in st.session_state.messages:
    if isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)

if prompt := st.chat_input("Ask Bella a question..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(
            get_response(
                prompt,
                st.session_state.messages[-10:],
                api_key,
                system_prompt,
                chapter_content
            )
        )
        st.session_state.messages.append(AIMessage(content=response))

    save_to_mongodb(
        user_id=user_id,
        conversation=[{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
        class_name=formatted_class,
        subject_name=formatted_subject,
        chapter_name=formatted_chapter,
    )
