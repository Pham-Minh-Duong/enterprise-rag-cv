from pydantic import BaseModel, Field
from typing import Optional, List

class ChatRequest(BaseModel):
    question: str = Field(..., description="Câu hỏi của người dùng gửi tới hệ thống RAG")
    session_id: Optional[str] = Field(None, description="Mã phiên hội thoại (tùy chọn)")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="Câu trả lời được sinh ra từ LLM (Ollama)")
    sources: Optional[List[str]] = Field(default=[], description="Danh sách tài liệu nguồn tham khảo")
    latency_ms: Optional[float] = Field(None, description="Thời gian xử lý tính bằng mili-giây")

class LoginRequest(BaseModel):
    username: str = Field(..., description="Tên đăng nhập hệ thống")
    password: str = Field(..., description="Mật khẩu người dùng")

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"