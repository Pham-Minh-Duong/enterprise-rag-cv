import os
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

def get_rag_chain(model_name: str = "llama3.2"):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    chroma_host = os.getenv("CHROMA_HOST", "localhost")
    chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
    
    client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    vector_db = Chroma(
        client=client,
        collection_name="personal_kb",
        embedding_function=embeddings
    )
    
    # Lấy top 3 kết quả tương đồng nhất
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    llm = Ollama(
        base_url=OLLAMA_BASE_URL,
        model=model_name
    )

    prompt_template = """Dựa vào dữ liệu ngữ cảnh dưới đây, hãy liệt kê tất cả thông tin/công nghệ được đề cập để trả lời câu hỏi.

Ngữ cảnh:
{context}

Câu hỏi: {input}

Trả lời (bằng tiếng Việt):"""

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "input"]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain