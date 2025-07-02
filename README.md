
# 🤖 Aap Ka Apna Multi Document RAG Chatbot

A lightweight, user-friendly **RAG (Retrieval-Augmented Generation)** based chatbot that answers your questions based on **uploaded documents** (PDF, CSV, PPTX). Built using **Streamlit**, **LangChain**, **FAISS**, and **Ollama (LLaMA3/DeepSeek)**.

---

## 📌 Features

- 🧠 Retrieval-Augmented Generation using LangChain.
- 🔍 Embedding and semantic search with FAISS.
- 🗃️ Accepts PDF, CSV, and PowerPoint files.
- 💬 Chat history is stored in a local SQLite database per user.
- 📦 Local embedding using `nomic-embed-text` via Ollama.
- 🧼 One-click **delete** of vector database or chat history.
- ✨ Simple Streamlit UI for intuitive interaction.

---

## 📁 File Structure

```plaintext
chatbot_rag.py                 # Main Streamlit app
chat_history_aapka_sathi.db    # SQLite DB to store chat history
vector_db/                     # FAISS Vector Store directory
```


## 🚀 Getting Started

1. Clone the Repository

```bash
  git clone https://github.com/your-username/aap-ka-apna-rag-chatbot.git
cd aap-ka-apna-rag-chatbot
```

2. Install Dependencies

```bash
 pip install -r requirements.txt
```
Make sure `streamlit`, `langchain`, `langchain-community`, `faiss-cpu`, `tiktoken`, `PyMuPDF`, and `ollama` related packages are included.

3. Start Ollama with Required Models

```bash
ollama run nomic-embed-text
ollama run llama3.2:latest
```

4. Run the Streamlit App

```bash
streamlit run chatbot_rag.py
```

## 📂 Supported File Formats

- `.pdf`
- `.csv`
- `.pptx`
These files are loaded, **chunked**, **tokenized**, **embedded**, and **stored** using **FAISS** to enable semantic search and contextual answers.

## 🧠 How It Works

1. User uploads a file via sidebar.

2. The file is parsed and chunked.

3. Each chunk is embedded using Ollama.

4. FAISS indexes the vectors for semantic retrieval.

5. User submits a question.

6. Relevant chunks are retrieved and sent as context to the LLM.

7. LLM (via Ollama) answers based on uploaded data.



## Screenshots

### RAG UI

![App Screenshot](https://github.com/sonu275981/Multi_document-rag-chatbot-streamlit-ollama/blob/main/ss-1.png?raw=true)

### File uploaded, Vector DB created 

![App Screenshot](https://github.com/sonu275981/Multi_document-rag-chatbot-streamlit-ollama/blob/main/ss-2.png?raw=true)

### Vector DB deleated successfully

![App Screenshot](https://github.com/sonu275981/Multi_document-rag-chatbot-streamlit-ollama/blob/main/ss-3.png?raw=true)



## License

This project is open-source and free to use under the MIT License.


## ✍️ Author

Developed by Sonu Chaurasia 👨‍💻
Inspired by real-world document search and Gen-AI assistant tools.


## 🙋‍♂️ Feedback & Contributions

Contributions, suggestions, and feedback are most welcome! Feel free to open issues or submit pull requests.
