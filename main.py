import json
import os
import re
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Thiếu biến môi trường GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

BASE_DIR = Path(__file__).parent

# Giới hạn đơn giản: tối đa 6 lượt phân tích / phút / IP
_hits: dict[str, list[float]] = {}


def too_many(ip: str) -> bool:
    now = time.time()
    arr = [t for t in _hits.get(ip, []) if now - t < 60]
    arr.append(now)
    _hits[ip] = arr
    return len(arr) > 6

client = genai.Client(api_key=GEMINI_API_KEY)


class ProductRequest(BaseModel):
    product_name: str
    product_url: str | None = None
    listed_price: str | float
    condition: str
    user_need: str


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "Server Python đang hoạt động bình thường!"}


@app.post("/api/analyze")
async def analyze_product(data: ProductRequest, request: Request):
    ip = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip() or (
        request.client.host if request.client else "unknown"
    )
    if too_many(ip):
        raise HTTPException(
            status_code=429, detail="Bạn thao tác quá nhanh, vui lòng thử lại sau 1 phút."
        )
    try:
        price_str = re.sub(r"[^\d]", "", str(data.listed_price))
        if not price_str:
            raise HTTPException(
                status_code=400, detail="Giá tiền không hợp lệ. Vui lòng nhập số!"
            )
        clean_price = float(price_str)

        # --- TÍNH NĂNG MỚI: Tính toán quy đổi ra số bữa ăn sáng (chuẩn 50k/bữa) ---
        breakfast_count = round(clean_price / 50000, 1)

        url_info = (
            f"- Link tham khảo sản phẩm: {data.product_url}"
            if data.product_url
            else "- Link sản phẩm: Không cung cấp"
        )

        # --- TÍNH NĂNG MỚI: Bổ sung 2 trường bóc tách giá trị vào Prompt ---
        prompt = f"""
        Bạn là một chuyên gia thẩm định giá, cố vấn mua sắm thông minh và tâm lý học hành vi.
        Hãy phân tích toàn diện món hàng sau:
        - Tên sản phẩm: {data.product_name}
        {url_info}
        - Giá người bán đưa ra: {clean_price:,.0f} VNĐ
        - Tình trạng: {data.condition}
        - Nhu cầu sử dụng: {data.user_need}

        Yêu cầu trả về BẮT BUỘC dạng JSON thuần túy (không chứa mã markdown ```json) có cấu trúc như sau:
        {{
            "real_value": "Khoảng giá trị thực tế ước tính (ví dụ: 8.500.000 - 9.000.000 VNĐ)",
            "equivalent_comparison": "Món đồ giá {clean_price:,.0f} VNĐ này tương đương với khoảng {breakfast_count} ngày tiền ăn sáng của bạn.",
            "value_trap": "Bóc tách bẫy giá trị (Ví dụ: Khoảng 40% giá trị nằm ở bao bì/quảng cáo KOL, công dụng thực tế chỉ tương đương sản phẩm 100.000 VNĐ).",
            "status": "Rẻ / Hợp lý / Cao hơn thực tế",
            "recommendation": "Nên mua ngay / Bỏ qua / Đợi hạ giá / Mặc cả thêm",
            "reasons": [
                "Lý do phân tích 1",
                "Lý do phân tích 2",
                "Lý do phân tích 3"
            ],
            "suggestions": [
                "Gợi ý / Mẹo kiểm tra khi mua (ví dụ: test camera, check kĩ pin, đàm phán giảm 300k...)",
                "Gợi ý lựa chọn thay thế (nếu có)"
            ],
            "buying_tips": "Lời khuyên mua sắm / Đánh giá tổng quan dành cho người dùng"
        }}
        """

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        raw_text = response.text.strip()
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```[a-zA-Z]*\n?", "", raw_text)
            raw_text = re.sub(r"\n?```$", "", raw_text)

        result_json = json.loads(raw_text)
        return result_json

    except HTTPException:
        raise
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500, detail="Lỗi định dạng dữ liệu từ AI. Hãy thử lại!"
        )
    except Exception as e:
        # --- TÍNH NĂNG MỚI: Sửa lỗi mã hóa UTF-8 trên Windows để không bị crash app ---
        error_msg = str(e).encode("utf-8", errors="replace").decode("utf-8")
        raise HTTPException(
            status_code=500, detail=f"Lỗi xử lý Server: {error_msg}"
        )