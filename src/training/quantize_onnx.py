import os
import time
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from onnxruntime.quantization import quantize_dynamic, QuantType
import mlflow

def export_and_quantize():
    # 1. Khởi tạo MLflow tracking experiment[cite: 3]
    mlflow.set_experiment("enterprise-rag-embedding-optimization")
    
    with mlflow.start_run(run_name="onnx-int8-quantization"):
        print("📥 Đang tải mô hình embedding gốc...")
        model_name = "paraphrase-multilingual-MiniLM-L12-v2"
        model = SentenceTransformer(model_name)
        
        # Tạo thư mục lưu trữ nếu chưa có
        output_dir = "models/embedding/1"
        os.makedirs(output_dir, exist_ok=True)
        onnx_path = os.path.join(output_dir, "model.onnx")
        int8_path = os.path.join(output_dir, "model_int8.onnx")
        
        # 2. Export sang ONNX[cite: 3]
        print("🔄 Đang export mô hình sang định dạng ONNX...")
        # Lấy underlying transformer model từ sentence_transformers
        transformer_model = model[0].auto_model
        transformer_model.eval()
        
        # Tạo dummy input tương thích với cấu trúc sentence-transformer
        dummy_input_ids = torch.randint(0, 1000, (1, 128), dtype=torch.long)
        dummy_attention_mask = torch.ones((1, 128), dtype=torch.long)
        
        torch.onnx.export(
            transformer_model,
            (dummy_input_ids, dummy_attention_mask),
            onnx_path,
            input_names=['input_ids', 'attention_mask'],
            output_names=['embeddings'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'seq_len'},
                'attention_mask': {0: 'batch_size', 1: 'seq_len'},
                'embeddings': {0: 'batch_size'}
            },
            opset_version=14
        )
        print(f"✅ Đã export ONNX thành công tại: {onnx_path}")
        
        # 3. Dynamic Quantization INT8
        print("⚙️ Đang thực hiện Quantization INT8...")
        quantize_dynamic(
            model_input=onnx_path,
            model_output=int8_path,
            weight_type=QuantType.QUInt8
        )
        print(f"🎉 Đã quantize thành công tại: {int8_path}")
        
        # 4. Đo lường kích thước và log vào MLflow
        original_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        quantized_size_mb = os.path.getsize(int8_path) / (1024 * 1024)
        
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("quantization", "dynamic_int8")
        mlflow.log_metric("original_size_mb", round(original_size_mb, 2))
        mlflow.log_metric("quantized_size_mb", round(quantized_size_mb, 2))
        mlflow.log_artifact(int8_path)
        
        # Xóa bản onnx gốc không nén để tiết kiệm dung lượng đĩa
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
            
        print(f"📊 Kích thước mô hình sau nén (INT8): {quantized_size_mb:.2f} MB (Giảm ~4 lần so với gốc)")

if __name__ == "__main__":
    export_and_quantize()