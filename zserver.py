import os
import shutil
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import các lớp cần thiết từ các file của bạn
from data_chunker import DocumentChunker
from new_metadata_extractor import MetadataExtractor
from storing import NodeStorageHandler  # <-- Import từ file storing.py

# ======================================================================================
# 1. KHỞI TẠO CÁC ĐỐI TƯỢNG CỐ ĐỊNH
# ======================================================================================
# Các đối tượng này sẽ được tạo một lần khi ứng dụng khởi động
storage_handler: NodeStorageHandler = None
query_engine = None

# Thiết lập FastAPI lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code này chạy KHI ỨNG DỤNG KHỞI ĐỘNG
    global storage_handler, query_engine
    
    print("=== KHỞI TẠO NODE STORAGE HANDLER VÀ QUERY ENGINE ===")
    
    # Khởi tạo handler
    storage_handler = NodeStorageHandler(collection_name="sailing_docs_api")
    
    # Tải index và các store hiện có.
    # Bằng cách truyền vào một list rỗng, hàm build_automerging_index (phiên bản đã sửa)
    # sẽ chỉ tải storage hiện có mà không thêm node mới.
    print("Đang tải index hiện có (nếu có)...")
    storage_handler.build_automerging_index(nodes=[])
    
    # Tạo query engine từ index đã tải
    print("Đang tạo query engine...")
    query_engine = storage_handler.create_query_engine()
    
    print("✅ Hệ thống đã sẵn sàng nhận yêu cầu!")
    
    yield
    
    # Code này chạy KHI ỨNG DỤNG DỪNG (nếu cần dọn dẹp)
    print("Ứng dụng đang tắt...")

# Khởi tạo ứng dụng FastAPI với lifespan
app = FastAPI(lifespan=lifespan)

# Khởi tạo các thành phần xử lý file
meta_extractor = MetadataExtractor()
document_chunker = DocumentChunker(meta_extractor=meta_extractor)

# Thiết lập thư mục upload và static
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ======================================================================================
# 2. ĐỊNH NGHĨA CÁC API ENDPOINTS
# ======================================================================================

# Định nghĩa cấu trúc cho yêu cầu query
class QueryRequest(BaseModel):
    query: str

@app.get("/", response_class=HTMLResponse)
def read_root():
    """Trang chủ hiển thị file index.html."""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Lỗi: Không tìm thấy file index.html trong thư mục /static</h1>"


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint để tải file lên, xử lý và thêm vào Vector Store.
    """
    allowed_ext = {".docx", ".doc", ".pdf", ".txt"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_ext:
        raise HTTPException(status_code=400, detail="Loại file không được hỗ trợ.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # BƯỚC 1: Chunk file thành các node
        print(f"Đang xử lý file: {file.filename}...")
        nodes = document_chunker.chunk_from_file(file_path)

        if not nodes:
            os.remove(file_path)
            raise HTTPException(status_code=500, detail="Không thể trích xuất nội dung từ file.")
        
        # BƯỚC 2: Thêm các node mới vào index hiện có
        print(f"Đã tạo {len(nodes)} nodes. Đang thêm vào index...")
        storage_handler.build_automerging_index(nodes=nodes)
        print("✅ Đã thêm các node mới vào index thành công!")

        return {
            "filename": file.filename, 
            "message": f"File processed and indexed successfully. Total nodes added: {len(nodes)}."
        }
    except Exception as e:
        # Ghi lại lỗi chi tiết ở server-side
        print(f"Lỗi khi xử lý file upload: {e}")
        raise HTTPException(status_code=500, detail=f"Đã xảy ra lỗi khi xử lý file: {str(e)}")
    finally:
        # Đóng file để giải phóng tài nguyên
        await file.close()


@app.post("/query")
async def handle_query(request: QueryRequest):
    """
    Endpoint để nhận câu hỏi và trả về câu trả lời từ Query Engine.
    """
    if query_engine is None:
        raise HTTPException(status_code=503, detail="Query engine chưa sẵn sàng. Vui lòng thử lại sau.")

    print(f"Nhận được câu hỏi: {request.query}")
    try:
        response = query_engine.query(request.query)
        
        # Trích xuất các thông tin cần thiết từ response
        answer = str(response)
        source_nodes = [
            {"text": node.get_text()[:500] + "...", "score": node.get_score()} 
            for node in response.source_nodes
        ]

        return {
            "answer": answer,
            "source_nodes": source_nodes
        }
    except Exception as e:
        print(f"Lỗi khi thực hiện query: {e}")
        raise HTTPException(status_code=500, detail=f"Đã xảy ra lỗi khi xử lý câu hỏi của bạn: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    # Chạy server với app object là 'app' từ file 'server'
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)