import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory, RunnablePassthrough
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain import hub

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken 
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import os
import shutil

st.title("_Aap ka Apna_ :blue[RAG] :sunglasses:") # title of the chatbot

embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
base_url = "http://localhost:11434"
model = "llama3.2:latest"  # deepseek-r1:1.5b, llama3.2:1b, gemma3:4b llama3.2:latest etc.

llm = ChatOllama(
    model = model,
    temperature = 0.8,
    num_predict = 2048,
)
def chunking_data(data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.split_documents(data)
    return chunks

def total_tokens(data):
    encoding = tiktoken.get_encoding("cl100k_base")  
    token_counts = [len(encoding.encode(chunk.page_content)) for chunk in data if chunk.page_content]
    return token_counts

# def create_vector_embedding(chunks):
#     vector_db = FAISS.from_documents(chunks, embeddings)
#     return vector_db

def create_vector_embedding(chunks):
    db_path = "vector_db"  # Define a local directory for FAISS index

    if os.path.exists(db_path):  # Load if already exists
        vector_db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    else:  # Create and save FAISS index
        vector_db = FAISS.from_documents(chunks, embeddings)
        vector_db.save_local(db_path)
    return vector_db

def retrieve_vector_embedding(query, vector_db):
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    docs = retriever.invoke(query)
    if not docs:
        return ["No relevant documents found."]
    return docs

def delete_vector_db():
    db_path = "vector_db"
    if os.path.exists(db_path):
        shutil.rmtree(db_path, ignore_errors=True)
        return "✅ Vector DB deleted successfully!"
    else:
        return "⚠️ Vector DB does not exist."

def format_docs(docs):
    return '\n\n'.join([docs.page_content for docs in docs])



# --------------------------------------
# create a function to get chat history from SQLite database
def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, connection = "sqlite:///chat_history_aapka_sathi.db")

# --------------------------------------

# sidebar content
# Using object notation  
# st.sidebar.image("https://pngimg.com/uploads/baseball/baseball_PNG19068.png", width=200)

st.sidebar.title("Chatbot  :red[Settings] :star-struck:")
st.sidebar.write("This is a simple chatbot that uses RAG (Retrieval-Augmented Generation) to answer questions based on uploaded documents.")

# add_selectbox = st.sidebar.selectbox(
#     "How would you like to be contacted?",
#     ("Email", "Home phone", "Mobile phone")
# )

uploaded_data = []

uploaded_file = st.sidebar.file_uploader("Choose a file ", type=["pdf", "csv", "pptx"])
if uploaded_file is not None:
    st.sidebar.write("filename:", uploaded_file.name)
    # Save the file to a temporary location
    temp_file_path = f"./temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getvalue()) # -  writing/saving the file to the temp location
        # Read PDF content
    with st.spinner("Reading File..."):
        if uploaded_file.type == "application/pdf":
            uploaded_data = PyMuPDFLoader(temp_file_path).load()
        elif uploaded_file.type == "text/csv":
            uploaded_data = CSVLoader(temp_file_path).load()
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
            uploaded_data = UnstructuredPowerPointLoader(temp_file_path).load()
       
     # Display extracted text
    # st.write(uploaded_data) # -  printing uploaded data
    with st.spinner("Chunking your data..."):
        chunks = chunking_data(uploaded_data)
        st.sidebar.write("Length:", len(chunks))

    with st.spinner("Calculating total tokens..."):
        count_tokens = total_tokens(chunks)
        st.sidebar.write("Total tokens:", sum(count_tokens))
        # st.write(chunks[0].page_content)  # uncomment if needed - it is output of 1st chunk of document
        
    with st.spinner("Creating vector embedding..."):
        vector_db = create_vector_embedding(chunks)
        st.sidebar.write("Vector DB Loaded Successfully!")

    os.remove(temp_file_path)
    
    
# --------------------------------------

# user = "sonu"
user = st.text_input('Enter your User (User chats are saved based on their username) :', 'sonu')

chat_history = get_session_history(user)
# chat_history.get_messages()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --------------------------------------

system_message = SystemMessagePromptTemplate.from_template("You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Don't try to make up an answer.")
human_message = HumanMessagePromptTemplate.from_template("{input}")

message = [system_message, MessagesPlaceholder('chat_history'), human_message]

chat = ChatPromptTemplate(message)

chat_chain = chat | llm | StrOutputParser()

# create a runnable with history and chain
runnable_with_msg_history = RunnableWithMessageHistory(chat_chain, get_session_history, input_messages_key= 'input', history_messages_key= 'chat_history')

# --------------------------------------


# def chat_with_llm(user, message):
#     return runnable_with_msg_history.stream(
#         {"input": message},
#         config={"configurable": {"session_id": user}}
#     )

with st.form("my_input_form"):
    user_input = st.text_area('Enter your Question')
    submit_button = st.form_submit_button("Submit")

if submit_button:
    with st.spinner("Retrieving vector embedding..."):
        # query = "What is the Candidate name?"
        docs = retrieve_vector_embedding(user_input, vector_db) # ---- using retrieve_vector_embedding method
        context = format_docs(docs)  # ---- using context method
        if isinstance(context, list) and context[0] == "No relevant documents found.":
            st.write("❌ No relevant documents found in vector DB.")
        else:
            #st.write("Docs:", context)   # ---- uncomment if needed - it is output of all chunk of document by context method
            response = runnable_with_msg_history.stream(
            {"input": f"{context}\n\n{user_input}"},
            config={"configurable": {"session_id": user}}
        )
        st.write("🤖 AI Response:", response)

if st.button('Clear History'):
    st.session_state.chat_history = []
    chat_history.clear()

if st.button("Delete Vector DB"):
    message = delete_vector_db()
    st.write(message) 

        
# st.write(chat_with_llm(user, user_input))
