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
import docx2txt
import PyPDF2
import io

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
DEFAULT_SYSTEM_PROMPT = """
You are an AI assistant specializing in insurance risk assessment. Your knowledge comes from the transcripts of IFA Academy course or webinar videos. Your role is to assist users by providing accurate and concise information based on these transcripts. Always strive to be helpful, clear, and informative.
"""

sys_prompt = """You are an AI assistant specializing in insurance risk assessment. Your knowledge comes exclusively from the transcripts of IFA Academy course or webinar videos. Your role is to assist users by providing accurate and concise information based solely on these transcripts. Do not use any information outside of the provided context. Always strive to be helpful, clear, and informative within the bounds of the given information. Select response only from given context. """

instruction = """
Context:
{context}

Given the following user query related to insurance risk assessment
Question:{question}

*Instructions:*
- Analyze the user's query carefully.
- Retrieve relevant information from the provided context, which contains excerpts from course or webinar video transcripts.
- Formulate a clear, concise, and accurate response based only on the retrieved information.
- Do not use any knowledge or information that is not present in the given context.
- If the query cannot be fully answered with the given context, acknowledge this limitation and provide the best possible answer with the available information or context and acknowledge this also.
- Use professional language appropriate for discussing insurance risk assessment topics.
- If clarification is needed, ask focused follow-up questions.
- When appropriate, suggest related topics the user might find helpful for further learning, but only if these topics are mentioned in the provided context.
- If asked about something not covered in the context, state that the information is not available in the current course materials.

"""

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
def retrieval_qa_chain(llm, vectorstore):
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

# Function to read the content of files
def read_file(uploaded_file):
    if uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docx2txt.process(uploaded_file)
    elif uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        text = ""
        for page in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page].extract_text()
        return text
    else:
        st.warning(f"Unsupported file format: {uploaded_file.type}")
        return ""

# Process documents and store vectors in Pinecone
def process_documents(uploaded_files):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                    model_kwargs={'device': 'cpu'})
    texts = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)

    for uploaded_file in uploaded_files:
        content = read_file(uploaded_file)
        if content:
            texts.extend(text_splitter.split_text(content))

    # Set the embedding dimension manually as HuggingFaceEmbeddings does not expose this directly
    embedding_dimension = 384  # All-MiniLM-L6-v2 has a 384-dimensional embedding

    # Check if the index already exists, otherwise create it
    if index_name not in [index['name'] for index in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=embedding_dimension,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )

    # Initialize vectorstore and upload documents using LangChain's Pinecone wrapper
    vectorstore = LangchainPinecone.from_texts(texts, embeddings, index_name=index_name)
    return vectorstore

# Load vectorstore from Pinecone index
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    
    # Use the correct method to load from an existing index
    vectorstore = LangchainPinecone.from_existing_index(index_name=index_name, embedding=embeddings)
    return vectorstore

def main():
    st.title("Mekosha-IFA-VIIT Project")

    # Initialize session state
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None

    # Check if Pinecone index exists
    index_exists = index_name in [index['name'] for index in pc.list_indexes()]

    if index_exists and st.session_state.qa_chain is None:
        # Load the vectorstore and set up the QA chain
        with st.spinner("Loading existing database..."):
            vectorstore = load_vectorstore()
            llm = load_llm()
            st.session_state.qa_chain = retrieval_qa_chain(llm, vectorstore)
        st.success("Database loaded and ready to query!")

    st.sidebar.title("Upload Documents")
    uploaded_files = st.sidebar.file_uploader("Choose files", type=["txt", "pdf", "docx"], accept_multiple_files=True)
    
    if uploaded_files and st.sidebar.button("Process Documents"):
        with st.spinner("Processing documents..."):
            vectorstore = process_documents(uploaded_files)
            llm = load_llm()
            st.session_state.qa_chain = retrieval_qa_chain(llm, vectorstore)
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
