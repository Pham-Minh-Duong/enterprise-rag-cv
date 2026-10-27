import os
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.services.document_loader import load_and_split_documents

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = "personal_kb"

def build_vector_db():
    print("🔄 [1/3] Đang nạp và cắt nhỏ tài liệu...")
    chunks = load_and_split_documents()
    
    if not chunks:
        print("❌ Dừng tiến trình do không phát hiện dữ liệu.")
        return None

    print("🔄 [2/3] Khởi tạo Embedding Model Đa Ngôn Ngữ...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    print(f"🔄 [3/3] Kết nối tới ChromaDB Server tại {CHROMA_HOST}:{CHROMA_PORT}...")
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    
    # Xóa collection cũ nếu có để ghi đè vector đa ngôn ngữ mới
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        collection_name=COLLECTION_NAME
    )
    
    print("🎉 Thành công! Dữ liệu đã được lưu lại với Vector Đa Ngôn Ngữ!")
    return vector_db

if __name__ == "__main__":
    build_vector_db()