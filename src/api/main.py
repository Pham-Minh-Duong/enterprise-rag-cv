from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import time
import os
import traceback

from src.api.schemas import ChatRequest, ChatResponse, LoginRequest, Token
from src.api.auth import create_access_token, get_current_user
from src.services.vector_db import build_vector_db
from src.services.rag_chain import get_rag_chain

# Khởi tạo Rate Limiter dựa trên IP của client[cite: 2]
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Enterprise RAG-CV API Gateway",
    description="Hệ thống RAG kết hợp Computer Vision production-ready",
    version="1.0.0"
)

# Đăng ký Rate Limiter vào ứng dụng FastAPI[cite: 2]
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thư mục lưu trữ tài liệu upload tạm thời
UPLOAD_DIR = "data/documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {
        "message": "Welcome to Enterprise RAG-CV API!",
        "docs": "/docs",
        "status": "running"
    }

@app.post("/login", response_model=Token, summary="Đăng nhập nhận JWT token demo")
def login(form_data: LoginRequest):
    # Hard-code user demo cho giai đoạn phát triển ban đầu theo tài liệu
    if form_data.username == "admin" and form_data.password == "admin123":
        # Tạo JWT access token thật thay vì trả chuỗi mock cứng
        access_token = create_access_token(data={"sub": form_data.username})
        return {"access_token": access_token, "token_type": "bearer"}
    raise HTTPException(status_code=400, detail="Sai tên đăng nhập hoặc mật khẩu (gợi ý: admin / admin123)")

@app.post("/ingest", summary="Upload tài liệu và Index vào ChromaDB")
@limiter.limit("30/minute") # Giới hạn 30 request/phút[cite: 1, 2]
async def ingest_document(request: Request, file: UploadFile = File(...), user: str = Depends(get_current_user)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Gọi vector_db index dữ liệu
    db = build_vector_db()
    if not db:
        raise HTTPException(status_code=500, detail="Lỗi khi index vector database.")

    return {
        "status": "success", 
        "filename": file.filename, 
        "message": "Đã ingest và lưu vector thành công!", 
        "processed_by": user
    }

@app.post("/chat", response_model=ChatResponse, summary="Hỏi đáp thông minh với RAG")
@limiter.limit("30/minute") # Giới hạn 30 request/phút[cite: 1, 2]
async def chat_endpoint(request: Request, body: ChatRequest, user: str = Depends(get_current_user)):
    start_time = time.time()
    try:
        rag_chain = get_rag_chain()
        response = rag_chain.invoke({"input": body.question})
        
        answer = response.get("answer", "Không có câu trả lời.")
        sources = list(set([doc.metadata.get("source", "unknown") for doc in response.get("context", [])]))
        
        latency = (time.time() - start_time) * 1000
        return ChatResponse(answer=answer, sources=sources, latency_ms=round(latency, 2))
    except Exception as e:
        # In lỗi ra terminal và trả về chi tiết lỗi cho Swagger dễ theo dõi
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống RAG: {str(e)}")