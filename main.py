# backend/app/main.py
import os
import time
from typing import List, Dict, Optional
import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
from paddleocr import PaddleOCR
import pytesseract
from PIL import Image
import io
import numpy as np

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow cross domain access for all domains
    allow_methods=["*"],  # allow all http methods
    allow_headers=["*"],  # allow all request header
)

# Initialize OCR engines
# paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
paddle_ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    det_model_dir=None,
    rec_model_dir=None,
    cls_model_dir=None,
    enable_mkldnn=False,
    use_tensorrt=False
)
tesseract_config = '--oem 3 --psm 6'

# Database setup
DB_NAME = "ocr_results.db"


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            engine TEXT,
            text_content TEXT,
            confidence REAL,
            processing_time REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


init_db()


class OCRResult(BaseModel):
    engine: str
    text: str
    confidence: Optional[float]
    processing_time: float


class ProcessResponse(BaseModel):
    results: List[OCRResult]
    file_metadata: dict


def pdf_to_images(file_bytes: bytes):
    """Convert PDF to list of images"""
    images = []
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page in doc:
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes()))
            images.append(np.array(img))
    except Exception as e:
        raise ValueError(f"PDF processing failed: {str(e)}")
    return images


async def process_image_file(file: UploadFile) :
    """Process uploaded image file"""
    file_bytes = await file.read()
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img_array = np.array(img)

        # make sure it is 3-channel image
        if len(img_array.shape) == 2:  # gray process
            img_array = np.stack((img_array,) * 3, axis=-1)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]

        return [img_array]
    except Exception as e:
        raise ValueError(f"image process failed: {str(e)}")

def run_paddleocr(images: List[np.ndarray]):
    """Run PaddleOCR on images with correct result extraction"""
    start_time = time.time()
    full_text = []
    confidences = []

    for img in images:
        try:
            # Convert to BGR format if needed
            if img.shape[2] == 3:  # RGB
                img = img[:, :, ::-1]  # RGB -> BGR

            # Run OCR
            result = paddle_ocr.ocr(img)

            # Debug: print raw result structure
            # print(f"Raw OCR result type: {type(result)}")  # 调试输出类型

            # Extract text and confidence from new format
            page_text = ""
            page_conf = []

            if result and isinstance(result, list):
                # new version format
                for page_result in result:
                    if isinstance(page_result, dict) and 'rec_texts' in page_result:
                        page_text += "\n".join(page_result['rec_texts']) + "\n"
                        page_conf.extend(page_result['rec_scores'])

            full_text.append(page_text.strip())
            confidences.append(np.mean(page_conf) if page_conf else 0.0)

        except Exception as e:
            print(f"PaddleOCR Process Error: {str(e)}", exc_info=True)
            full_text.append("")
            confidences.append(0.0)

    return OCRResult(
        engine="PaddleOCR",
        text="\n\n".join(full_text),
        confidence=np.mean(confidences) if confidences else None,
        processing_time=(time.time() - start_time) * 1000
    )


def run_tesseract(images: List[np.ndarray]):
    """Run Tesseract OCR on images with confidence"""
    start_time = time.time()
    full_text = []
    confidences = []

    for img in images:
        pil_img = Image.fromarray(img)

        # 获取完整OCR数据（包含置信度）
        data = pytesseract.image_to_data(
            pil_img,
            config=tesseract_config,
            output_type=pytesseract.Output.DICT
        )

        # 提取文本和置信度（处理浮点数字符串）
        text_lines = []
        page_confs = []

        for i, text in enumerate(data['text']):
            if text.strip():  # 只处理非空文本
                try:
                    # 先将字符串转为float，再转为int（处理96.721733这种情况）
                    conf = float(data['conf'][i])
                    if conf > 0:  # 过滤无效置信度
                        page_confs.append(conf / 100)  # 转换为0-1范围
                        text_lines.append(text)
                except (ValueError, TypeError):
                    continue

        full_text.append(" ".join(text_lines))
        confidences.append(page_confs)

    # 计算整体平均置信度
    all_confs = [c for conf_list in confidences for c in conf_list]
    overall_confidence = np.mean(all_confs) if all_confs else None

    return OCRResult(
        engine="Tesseract",
        text="\n\n".join(full_text),
        confidence=overall_confidence,
        processing_time=(time.time() - start_time) * 1000
    )


def save_to_db(filename: str, result: OCRResult):
    """Save results to database"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO results (filename, engine, text_content, confidence, processing_time)
        VALUES (?, ?, ?, ?, ?)
    ''', (filename, result.engine, result.text[:10000], result.confidence, result.processing_time))
    conn.commit()
    conn.close()


@app.post("/process", response_model=ProcessResponse)
async def process_file(file: UploadFile = File(...)):
    """Main processing endpoint"""
    try:
        # Determine file type and convert to images
        if file.filename.endswith('.pdf'):
            images = pdf_to_images(await file.read())
        else:
            images = await process_image_file(file)

        # Process with both engines
        results = [
            run_paddleocr(images),
            run_tesseract(images)
        ]

        # Save results to DB
        for result in results:
            save_to_db(file.filename, result)

        return {
            "results": results,
            "file_metadata": {
                "filename": file.filename,
                "pages": len(images),
                "size": file.size
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/search")
async def search_results(query: str = "", limit: int = 10):
    """Search previous results"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    if query:
        cursor.execute('''
            SELECT * FROM results 
            WHERE text_content LIKE ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (f"%{query}%", limit))
    else:
        cursor.execute('''
            SELECT * FROM results 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))

    results = cursor.fetchall()
    conn.close()

    return {
        "count": len(results),
        "results": [{
            "id": r[0],
            "filename": r[1],
            "engine": r[2],
            "text_snippet": r[3][:300],
            "confidence": r[4],
            "processing_time": r[5],
            "timestamp": r[6]
        } for r in results]
    }