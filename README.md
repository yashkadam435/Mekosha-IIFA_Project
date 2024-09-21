# Mekhosha-IFA-VIIT Project

This project provides a question-answering application using the LangChain framework and Streamlit. It allows users to upload text documents, process them into a searchable knowledge base, and query them for relevant answers using a retrieval-based system. The system is powered by a Large Language Model (LLM), using Llama-2 and HuggingFace embeddings, and supported by FAISS for efficient document retrieval.

## Features

- Upload multiple `.txt` files and process them.
- Query the uploaded documents using a retrieval-augmented question-answering system.
- Supports embedding generation using `sentence-transformers/all-MiniLM-L6-v2`.
- Uses FAISS for similarity search.
- LLM-based answer generation using Llama-3.
- Retrieves and displays source documents along with answers.

## Requirements

Make sure you have Python 3.8 or higher installed. The required dependencies can be installed using the following:

```bash
pip install -r requirements.txt
```

## Requirements.txt:

- streamlit
- python-dotenv
- langchain
- PyPDF2
- faiss-cpu
- langchain-community
- langchain-groq
- langchain-huggingface

## Usage
1. Clone the Repository

```bash
git clone https://github.com/your-username/Mekhosha-IFA-VIIT.git
cd Mekhosha-IFA-VIIT
```

## 2. Set up Environment Variables

Create a .env file in the root directory with the following keys:

```bash
GROQ_API_KEY=<your_groq_api_key>
```

## 3. Run the Application

You can start the Streamlit app by running:

```bash
streamlit run app.py
```

### 4. Upload Documents and Query

- Upload .txt files from the sidebar.
- Submit your query in the input box to receive answers along with the relevant source documents.

## Project Structure

```bash
.
├── app.py              # Main application code
├── requirements.txt    # Required packages
└── README.md           # Project documentation

```

## How It Works

1) Document Processing:

- Upload your text files. The system splits the documents into smaller chunks using a RecursiveCharacterTextSplitter.
- FAISS is used to store embeddings of these chunks for fast retrieval based on semantic similarity.

2) LLM Response Generation:

- A question-answering chain is built using the RetrievalQA from LangChain.
- The selected LLM (Llama-2) is used to generate answers based on the retrieved context from the FAISS store.

3) Query Submission:

- Users can input their queries. The application fetches relevant document chunks and generates an answer using the LLM.
- The response and related source documents are displayed.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
