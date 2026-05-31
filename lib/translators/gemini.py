"""
Gemini-based translation backend.

This file only contains Gemini-specific logic.
Caching and batching are handled by lib.translators.base.
"""

import json
from typing import Dict, List

import google.generativeai as genai


def create_gemini_model(
    api_key: str,
    model_name: str = "models/gemini-2.5-pro",
):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def build_vi_to_han_prompt(terms: List[str]) -> str:
    return f"""
Bạn là một học giả chuyên về Lịch sử Việt Nam trung đại và văn tự Hán cổ.

Nhiệm vụ:
Chuyển các từ/cụm từ Chữ Quốc Ngữ sang Hán tự Phồn Thể/Hán văn lịch sử tương ứng.

YÊU CẦU BẮT BUỘC:
1. Tuyệt đối không dịch theo nghĩa tiếng Trung hiện đại nếu từ đó là tên riêng lịch sử Việt Nam.
2. Mỗi chuỗi có dấu gạch dưới "_" là một đơn vị nguyên khối, phải xét toàn bộ cụm trước khi chuyển sang Hán tự.
3. Với các cụm có từ loại/phụ tố như "nhà", "kinh_thành", "sông", "núi", chỉ giữ phần tên riêng cốt lõi nếu phụ tố đó không thuộc tên Hán chính thức.
   Ví dụ:
   - "nhà_Tần" -> "秦"
   - "kinh_thành_Thăng_Long" -> "昇龍"
4. Ưu tiên dạng Hán văn lịch sử đã dùng trong văn hiến Việt Nam nếu có.
5. Chỉ dùng chữ Phồn thể / Hán văn cổ. Không dùng giản thể.
6. Không giải thích, không chú thích, không thêm văn bản ngoài JSON.
7. Nếu không chắc chắn hoặc không xác định được, trả về chuỗi rỗng "" cho mục đó.
8. Không được bịa ra đáp án.

VÍ DỤ CHUẨN:

Input: ["Kinh_Dương_Vương", "Âu_Cơ", "Giao_Chỉ", "nhà_Tần", "Đại_Việt"]
Output: {{"Kinh_Dương_Vương": "涇陽王", "Âu_Cơ": "嫗姬", "Giao_Chỉ": "交阯", "nhà_Tần": "秦", "Đại_Việt": "大越"}}

Input: ["Sơn_Tinh", "Nhâm_Ngao", "Triệu_Đà", "Hoàng_Khê"]
Output: {{"Sơn_Tinh": "山精", "Nhâm_Ngao": "任囂", "Triệu_Đà": "趙佗", "Hoàng_Khê": "湟谿"}}

BÂY GIỜ HÃY DỊCH DANH SÁCH SAU:

Input: {json.dumps(terms, ensure_ascii=False)}

Chỉ trả về đúng một JSON object dạng:
{{
  "Từ_Quốc_Ngữ": "Từ_Hán_Phồn_Thể"
}}
""".strip()


def build_han_to_vi_prompt(terms: List[str]) -> str:
    return f"""
Bạn là một học giả chuyên về Lịch sử Việt Nam trung đại và văn tự Hán cổ.

Nhiệm vụ:
Chuyển các từ/cụm từ Hán tự (Phồn Thể / Hán văn lịch sử) sang Chữ Quốc Ngữ tương ứng,
ưu tiên cách đọc/cách ghi đã dùng trong sử Việt Nam.

YÊU CẦU BẮT BUỘC:
1. Với tên riêng lịch sử Việt Nam, dùng đúng tên Quốc Ngữ đã quen dùng trong sử Việt,
   KHÔNG phiên âm theo âm Hán hiện đại của Trung Quốc.
   Ví dụ: "涇陽王" -> "Kinh Dương Vương" (không phải "Kính Dương Vương" kiểu khác).
2. Trả về dạng có dấu, viết hoa đúng chuẩn tên riêng, dùng dấu cách giữa các âm tiết
   (KHÔNG dùng dấu gạch dưới "_").
3. Nếu là danh từ chung, trả về nghĩa tiếng Việt thông dụng nhất.
4. Không giải thích, không chú thích, không thêm văn bản ngoài JSON.
5. Nếu không chắc chắn hoặc không xác định được, trả về chuỗi rỗng "" cho mục đó.
6. Không được bịa ra đáp án.

VÍ DỤ CHUẨN:

Input: ["涇陽王", "嫗姬", "交阯", "秦", "大越"]
Output: {{"涇陽王": "Kinh Dương Vương", "嫗姬": "Âu Cơ", "交阯": "Giao Chỉ", "秦": "nhà Tần", "大越": "Đại Việt"}}

Input: ["山精", "任囂", "趙佗", "湟谿"]
Output: {{"山精": "Sơn Tinh", "任囂": "Nhâm Ngao", "趙佗": "Triệu Đà", "湟谿": "Hoàng Khê"}}

BÂY GIỜ HÃY DỊCH DANH SÁCH SAU:

Input: {json.dumps(terms, ensure_ascii=False)}

Chỉ trả về đúng một JSON object dạng:
{{
  "Từ_Hán": "Từ Quốc Ngữ"
}}
""".strip()


def _translate_with_gemini(
    model,
    terms: List[str],
    prompt_builder,
    verbose: bool = False,
) -> Dict[str, str]:
    """
    Generic single-direction Gemini translation.

    Signature compatible with BatchTranslator after binding model + prompt:
        List[str] -> Dict[str, str]
    """
    if not terms:
        return {}

    clean_terms: List[str] = []
    seen = set()

    for term in terms:
        term = str(term).strip()
        if not term or term in seen:
            continue
        seen.add(term)
        clean_terms.append(term)

    if not clean_terms:
        return {}

    prompt = prompt_builder(clean_terms)

    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.0,
                "response_mime_type": "application/json",
            },
        )

        raw_result = json.loads(response.text)

        result: Dict[str, str] = {}
        for term in clean_terms:
            value = raw_result.get(term, "")
            result[term] = str(value).strip() if value else ""

        if verbose:
            hit = sum(1 for value in result.values() if value)
            missing = [key for key, value in result.items() if not value]
            print(f"Gemini translated {hit}/{len(clean_terms)} terms")
            if missing:
                print(f"Empty translations: {missing}")

        return result

    except json.JSONDecodeError:
        if verbose:
            print(f"Gemini did not return valid JSON for terms: {clean_terms}")
        return {}

    except Exception as exc:
        if verbose:
            print(f"Gemini translation API error: {exc}")
        return {}


# Prompt builder per translation direction.
_PROMPT_BUILDERS = {
    ("vi", "zh"): build_vi_to_han_prompt,
    ("zh", "vi"): build_han_to_vi_prompt,
}


def create_gemini_translate_fn(
    api_key: str,
    model_name: str = "models/gemini-2.5-pro",
    src_lang: str = "vi",
    tgt_lang: str = "zh",
    verbose: bool = False,
):
    """
    Create a translate_fn compatible with BatchTranslator.

    Supported directions: vi->zh and zh->vi.

    Returns:
        Callable[[List[str]], Dict[str, str]]
    """
    prompt_builder = _PROMPT_BUILDERS.get((src_lang, tgt_lang))
    if prompt_builder is None:
        raise NotImplementedError(
            f"Gemini translation is not implemented for {src_lang}->{tgt_lang}"
        )

    model = create_gemini_model(api_key=api_key, model_name=model_name)

    def translate_fn(terms: List[str]) -> Dict[str, str]:
        return _translate_with_gemini(
            model=model,
            terms=terms,
            prompt_builder=prompt_builder,
            verbose=verbose,
        )

    return translate_fn
