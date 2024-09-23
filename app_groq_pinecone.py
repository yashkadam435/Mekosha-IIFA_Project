import os
import streamlit as st
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
import warnings
from langchain_groq import ChatGroq

# Load environment variables and configure API keys
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")

# Initialize Pinecone using the new Pinecone class approach
pc = Pinecone(api_key=pinecone_api_key)

# Define Pinecone index name
index_name = "mekhosha-ifa-viit-project"

# Suppress warnings
warnings.filterwarnings("ignore")

# LLaMA-2 prompt style and system prompts
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

sys_prompt = """Select response only from stored database. You are an expert assistant tasked with answering questions based on the provided context about webinar transcripts."""

instruction = """CONTEXT:\n\n {context}\n

Question: {question}"""

def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

prompt_template = get_prompt(instruction, sys_prompt)

llama_prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": llama_prompt}

# Function to create the RetrievalQA chain using Pinecone
def retrieval_qa_chain(llm, prompt, vectorstore):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.3, "k": 4}),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )
    return qa_chain

def load_llm():
    return ChatGroq(temperature=0.4, model_name="llama3-8b-8192")

# Process documents and store vectors in Pinecone
def process_documents(uploaded_files):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    texts = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)

    for uploaded_file in uploaded_files:
        content = uploaded_file.read().decode("utf-8")
        texts.extend(text_splitter.split_text(content))

    # Set the embedding dimension manually as HuggingFaceEmbeddings does not expose this directly
    embedding_dimension = 384  # All-MiniLM-L6-v2 has a 384-dimensional embedding

    # Check if the index already exists, otherwise create it
    if index_name not in [index.name for index in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=embedding_dimension,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )

    # Initialize vectorstore and upload documents using LangChain's Pinecone wrapper
    vectorstore = LangchainPinecone.from_texts(texts, embeddings, index_name=index_name)
    return vectorstore

def main():
    st.title("Mekhosha-IFA-VIIT Project")

    # Initialize session state
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None

    st.sidebar.title("Upload TXT Document")
    uploaded_files = st.sidebar.file_uploader("Choose files", type=["txt"], accept_multiple_files=True)
    
    if uploaded_files and st.sidebar.button("Process Documents"):
        with st.spinner("Processing documents..."):
            vectorstore = process_documents(uploaded_files)
            llm = load_llm()
            st.session_state.qa_chain = retrieval_qa_chain(llm, llama_prompt, vectorstore)
        st.sidebar.success("Documents processed!")

    # Query input and response
    query = st.text_input("Enter your query:")
    if st.button("Submit") and query:
        if st.session_state.qa_chain:
            with st.spinner("Searching for answers..."):
                response = st.session_state.qa_chain({'query': query})
                answer = response["result"]
                source_documents = response["source_documents"]
                
                # If no sources, return "I don't know the answer."
                if not source_documents:
                    st.write("I don't know the answer.")
                else:
                    st.success("Found an answer!")
                    st.markdown("#### Answer:")
                    st.write(answer)
                    st.markdown("#### Source Documents:")
                    for doc in source_documents:
                        st.write(doc)
        else:
            st.warning("Please process documents before submitting a query.")

if __name__ == "__main__":
    main()
