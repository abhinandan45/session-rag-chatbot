import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # <--- FIX: Corrected import location
from langchain_community.vectorstores import Chroma
from langchain_community.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()
# Note: GEMINI_API_KEY ko os.getenv() se fetch karna surakshit hai
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file.")

# --- 1. SETUP (Models are global) ---
# Embeddings: Documents ko vectors mein badalne ke liye
EMBEDDING_MODEL = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004", 
    google_api_key=GEMINI_API_KEY
) 
# LLM: User ke sawaalon ka jawab dene ke liye
LLM = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.1,
    google_api_key=GEMINI_API_KEY
)
BASE_DB_PATH = "data/vector_stores" # Ab hum multiple stores rakhenge

class RAGSystem:
    # Ab __init__ mein session_id lenge
    def __init__(self, session_name: str):
        self.session_name = session_name
        # Har session ke liye alag database path
        self.db_path = os.path.join(BASE_DB_PATH, self.session_name)
        self.vector_store = None
        
        # General-Purpose Prompt - wahi rakhenge jo humne fix kiya tha
        custom_prompt = """
        You are a highly professional, accurate, and specialized knowledge assistant. 
        Your primary goal is to answer the user's question ONLY and STRICTLY based on the context provided below.
        
        The context is extracted from the document(s) loaded for this specific session.
        You must analyze all structured data (like tables or lists) in the context to find the exact details.
        
        If the exact answer, specific data, or rule is not found in the context, clearly state: 
        "Based on the provided documents, I could not find the exact answer or data required."
        
        Answer in a professional, human-like, and neutral tone.
        
        Context:
        {context}
        
        Question:
        {question}
        
        Answer:
        """
        self.PROMPT = PromptTemplate(template=custom_prompt, input_variables=["context", "question"])

        # Vector store ko load/initialize karo
        self._load_vector_store()
    
    def _load_vector_store(self):
        """Loads an existing ChromaDB instance for this session or initializes a new one."""
        try:
            # New Session ke liye ya Existing Session ke liye load karega
            self.vector_store = Chroma(
                persist_directory=self.db_path,
                embedding_function=EMBEDDING_MODEL
            )
            print(f"ChromaDB loaded for session: {self.session_name}")
        except Exception:
            # Agar folder exist nahi karta, toh empty store banao
            os.makedirs(self.db_path, exist_ok=True)
            self.vector_store = Chroma.from_texts(
                texts=[" "], # Placeholder text
                embedding=EMBEDDING_MODEL,
                persist_directory=self.db_path
            )
            print(f"New ChromaDB initialized for session: {self.session_name}")
            
        # RetrievalQA Chain ko ab initialize karte hain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=LLM,
            chain_type="stuff",
            # k=3: matlab sabse zyada relevant 3 chunks hi retrieve honge
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": self.PROMPT},
            return_source_documents=False
        )
        
    def ingest_document(self, file_path: str):
        """PDF file ko load karke chunks mein divide karta hai aur vector store mein dalta hai."""
        print(f"Loading document: {file_path} into session {self.session_name}...")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Document ko chunks mein todne ke liye
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        texts = text_splitter.split_documents(documents)

        # Vector Store ko update karo
        self.vector_store.add_documents(texts)
        self.vector_store.persist() # Disk par save karo
        print(f"Document successfully added and DB persisted for session: {self.session_name}.")
        
    def query(self, question: str) -> str:
        """Vector store se relevant context nikalta hai aur LLM se jawab generate karwata hai."""
        if not self.vector_store:
            return "Error: Vector store is not initialized."
        
        result = self.qa_chain.invoke({"query": question})
        return result['result']

# Ab hum global 'rag_system' object nahi banayenge! Uski jagah app.py mein banayenge.
