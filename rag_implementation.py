from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

class RAGSystem:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", 
                 chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize the RAG system with embeddings and text splitter."""
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        self.vector_store = None
        self.retriever = None
        self.llm = None

    def load_documents(self, file_path: str, file_type: str = None):
        """Load documents from a file or directory."""
        if os.path.isdir(file_path):
            if file_type == "pdf":
                loader = DirectoryLoader(file_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
            else:
                loader = DirectoryLoader(file_path, glob="**/*.txt", loader_cls=TextLoader)
        else:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path)
        
        return loader.load()

    def process_documents(self, documents):
        """Process and split documents into chunks."""
        return self.text_splitter.split_documents(documents)

    def create_vector_store(self, documents):
        """Create a FAISS vector store from documents."""
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        return self.vector_store

    def set_llm(self, model_repo: str = "google/flan-t5-base"):
        """Set up the language model for generation."""
        load_dotenv()
        openai_model = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.5,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.llm = openai_model

    def setup_rag_chain(self):
        """Set up the RAG chain with prompt template."""
        if not self.retriever or not self.llm:
            raise ValueError("Vector store and LLM must be initialized first")

        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return self.rag_chain

    def query(self, question: str) -> dict:
        """Query the RAG system."""
        if not hasattr(self, 'rag_chain'):
            raise ValueError("RAG chain not set up. Call setup_rag_chain() first.")

        docs = self.retriever.get_relevant_documents(question)

        # Format the sources
        sources = [{
            "content": doc.page_content,
            "metadata": doc.metadata
        } for doc in docs]

        answer = self.rag_chain.invoke(question)

        return {
            "answer": answer,
            "sources": sources
        }

def main():
    # Example usage
    rag = RAGSystem()
    
    # Load and process documents
    documents = rag.load_documents("./documents", "pdf")
    processed_docs = rag.process_documents(documents)

    # Create vector store
    vector_store = rag.create_vector_store(processed_docs)
    
    # Set up LLM (requires HUGGINGFACEHUB_API_KEY in environment)
    rag.set_llm()
    
    # Set up RAG chain
    rag.setup_rag_chain()
    
    # Example query
    while True:
        question = input("\nAsk a question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        
        try:
            result = rag.query(question)
            print(f"\nAnswer: {result['answer']}")

            if result['sources']:
                for i, source in enumerate(result['sources'], 1):
                    print(f"\nSource {i}:")
                    print(f"Content: {source['content'][:200]}...")  # Show first 200 chars
                    print(f"Source: {source['metadata'].get('source', 'Unknown')}")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
