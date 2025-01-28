import os
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from pymongo import MongoClient
from uuid import uuid4  # For unique user IDs

mongo_client = MongoClient(st.secrets["MONGODB_URI"])  # Securely use the URI from secrets
db = mongo_client["bella"]  # Database name
conversations_collection = db["conversations"]  # Collection name

llm = None

def format_name(name):
    """Format filenames into readable titles"""
    return name.replace('_', ' ').replace('.txt', '').title()

def get_llm_instance(api_key):
    global llm
    if llm is None:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.3,
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
    """Save conversation to MongoDB."""
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

# Streamlit app setup
st.set_page_config(page_title="Bella", page_icon=":robot:")
st.title("বেলা::Bella")

# Sidebar controls
api_key = st.secrets["GEMINI_API_KEY"]

# File selection
base_dir = "./books/nine/"
selected_class = st.sidebar.selectbox("Select Class", os.listdir(base_dir))
selected_subject = st.sidebar.selectbox(
    "Select Subject",
    os.listdir(os.path.join(base_dir, selected_class))
)
selected_chapter = st.sidebar.selectbox(
    "Select Chapter",
    os.listdir(os.path.join(base_dir, selected_class, selected_subject))
)

# Format names for display
formatted_class = format_name(selected_class)
formatted_subject = format_name(selected_subject)
formatted_chapter = format_name(selected_chapter)

# User session management
if "user_id" not in st.session_state:
    st.session_state["user_id"] = str(uuid4())  # Generate a unique ID for the user

user_id = st.session_state["user_id"]
current_selection = (selected_class, selected_subject, selected_chapter)

# Initialize messages
if "messages" not in st.session_state:
    greeting = f"Hey! I'm Bella. Ready to explore chapter **{formatted_chapter}** in **{formatted_subject}** for class **{formatted_class}**?"
    st.session_state.messages = [AIMessage(content=greeting)]

# Load chapter content
chapter_path = os.path.join(base_dir, selected_class, selected_subject, selected_chapter)
with open(chapter_path, "r", encoding="utf-8") as file:
    chapter_content = file.read()

# Load system prompt
with open("bella_system.md", "r") as file:
    system_prompt = file.read().format(
        class_name=formatted_class,
        subject=formatted_subject,
        chapter=formatted_chapter,
        content=chapter_content
    )

# Display chat history
for message in st.session_state.messages:
    if isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)

# Chat input
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

    # Save the conversation to MongoDB
    save_to_mongodb(
        user_id=user_id,
        conversation=[{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
        class_name=formatted_class,
        subject_name=formatted_subject,
        chapter_name=formatted_chapter,
    )
