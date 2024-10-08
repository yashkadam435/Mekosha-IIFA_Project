import streamlit as st
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import warnings
import os

# Load environment variables and configure API keys
load_dotenv()
# os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

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
- If the query cannot be fully answered with the given context, acknowledge this limitation and provide the best possible answer with the available information or context and aacknowledge this also.
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

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold":0.3,"k": 4}),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )
    return qa_chain

def load_llm():
    return ChatGroq(temperature=0.4, model_name="llama3-8b-8192")

def process_documents(uploaded_files):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    texts = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)

    for uploaded_file in uploaded_files:
        content = uploaded_file.read().decode("utf-8")
        texts.extend(text_splitter.split_text(content))

    return FAISS.from_texts(texts, embeddings)

def main():
    st.title("Mekhosha-IFA-VIIT Project")

    # Initialize session state
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None

    st.sidebar.title("Upload TXT Document")
    uploaded_files = st.sidebar.file_uploader("Choose files", type=["txt"], accept_multiple_files=True)
    
    if uploaded_files and st.sidebar.button("Process Documents"):
        with st.spinner("Processing documents..."):
            db = process_documents(uploaded_files)
            llm = load_llm()
            st.session_state.qa_chain = retrieval_qa_chain(llm, llama_prompt, db)
        st.sidebar.success("Documents processed!")

    # Query input and response
    query = st.text_input("Enter your query:")
    if st.button("Submit") and query:
        if st.session_state.qa_chain:
            with st.spinner("Searching for answers..."):
                response = st.session_state.qa_chain({'query': query})
                st.success("Found an answer!")
                st.markdown("#### Answer:")
                st.write(response["result"])
                st.markdown("#### Source Documents:")
                st.write(response["source_documents"])
        else:
            st.warning("Please process documents before submitting a query.")

if __name__ == "__main__":
    main()
