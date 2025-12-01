import json
import os
import re
import sys
from typing import List, Dict, Any

import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModel

sys.stdout.reconfigure(encoding="utf-8")


# 전역 설정 / 파일 로딩

# 실행 경로 기준 디렉터리
if "__file__" in globals():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    BASE_DIR = os.getcwd()

# patterns.json 로딩
with open(os.path.join(BASE_DIR, "patterns.json"), "r", encoding="utf-8") as f:
    PATTERN_CONFIG = json.load(f)

# knowledge_base.json 로딩
with open(os.path.join(BASE_DIR, "knowledge_base.json"), "r", encoding="utf-8") as f:
    KNOWLEDGE_BASE = json.load(f)

DETECTION_THRESHOLDS = PATTERN_CONFIG.get("detection_thresholds", {})
MIN_CONF = DETECTION_THRESHOLDS.get("min_confidence", 0.7)

# NER 파이프라인 (KoELECTRA)

NER_MODEL_NAME = "monologg/koelectra-base-finetuned-naver-ner"

ner_pipeline = pipeline(
    "token-classification",
    model=NER_MODEL_NAME,
    tokenizer=NER_MODEL_NAME,
    aggregation_strategy="simple"  # WordPiece 조각 합침
)

# BERT 임베딩 모델 (beomi/KcBERT-base)

BERT_MODEL_NAME = "beomi/KcBERT-base"

bert_tok = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME)

bert_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(bert_device)
bert_model.eval()

print(f"Device set to use {bert_device}")


# 공통 유틸

def preprocess_text(text: str) -> str:
    """양끝 공백 제거 + 연속 공백 하나로 축소."""
    text = re.sub(r"\s+", " ", text.strip())
    return text


def summarize_ner(ner_results: List[Dict[str, Any]],
                  max_items: int = 5,
                  min_score: float = 0.0) -> List[str]:
    summary = []
    for item in ner_results:
        word = item.get("word", "")
        if word.startswith("##"):
            continue
        if re.fullmatch(r"[^\w가-힣]+", word or ""):
            continue

        score = float(item.get("score", 0.0))
        if score < min_score:
            continue

        entity = item.get("entity_group") or item.get("entity") or "UNK"
        summary.append(f"{word}({entity}, {score:.2f})")

    return summary[:max_items]


def summarize_candidates(candidates: List[Dict[str, Any]],
                         max_items: int = 5) -> List[str]:
    sorted_cands = sorted(
        candidates,
        key=lambda x: x.get("similarity", 0.0),
        reverse=True
    )
    summary = []
    for c in sorted_cands[:max_items]:
        text = c.get("text", "")
        source = c.get("source", "?")
        sim = c.get("similarity", 0.0)
        summary.append(f"{text} [{source}], sim={sim:.2f}")
    return summary


def format_final_entity(entity: Dict[str, Any] | None) -> str:
    if entity is None:
        return "None"

    text = entity.get("text", "")
    category = entity.get("category", "")
    source = entity.get("source", "")
    sim = entity.get("similarity", 0.0)
    return f"{text} (category={category}, source={source}, sim={sim:.2f})"


# 1단계: 규칙 기반 탐지 (patterns.json + knowledge_base.json)

def detect_expressions(text: str) -> List[Dict[str, Any]]:
    """
    patterns.json의 패턴 + knowledge_base.json의 alias를 이용한 규칙 기반 탐지.
    반환: 각 매칭에 대한 dict 리스트.
    """
    detected: List[Dict[str, Any]] = []

    # 1-1) patterns.json에 정의된 정규표현식 기반 탐지
    for pattern_name, info in PATTERN_CONFIG.get("patterns", {}).items():
        regex = info.get("regex")
        if not regex:
            continue
        conf = info.get("confidence", 1.0)
        compiled = re.compile(regex)

        for m in compiled.finditer(text):
            span_text = m.group(0)
            item = {
                "category": pattern_name,
                "pattern_name": pattern_name,
                "keyword": span_text,
                "start": m.start(),
                "end": m.end(),
                "text": span_text,
                "confidence": conf,
            }

            # 예: company_anonymized / initial_company 등에 대해
            # entity_candidates가 있으면 후보를 추가로 저장
            entity_cands = info.get("entity_candidates", {})
            if entity_cands:
                # 완전히 같은 키가 있으면 그걸 사용 (예: S사, L전자 등)
                key = span_text.replace(" ", "")
                if key in entity_cands:
                    item["entity_short_candidates"] = entity_cands[key]

            detected.append(item)

    # 1-2) knowledge_base.json의 aliases 기반 탐지
    for kb_section, entities in KNOWLEDGE_BASE.items():
        for canonical_name, meta in entities.items():
            aliases = meta.get("aliases", [])
            alias_list = set(aliases + [canonical_name])

            for alias in alias_list:
                if not alias:
                    continue
                for m in re.finditer(re.escape(alias), text):
                    detected.append(
                        {
                            "category": kb_section,
                            "pattern_name": f"kb_{kb_section}",
                            "keyword": alias,
                            "start": m.start(),
                            "end": m.end(),
                            "text": m.group(0),
                            "confidence": 0.95,
                            "kb_section": kb_section,
                            "kb_key": canonical_name,
                        }
                    )

    return detected

# 2단계: KoELECTRA NER

def extract_entities(text: str) -> List[Dict[str, Any]]:
    return ner_pipeline(text)


# 3단계: 후보군 생성 + BERT 임베딩 기반 코사인 유사도

def get_kb_metadata(section: str, key: str) -> Dict[str, Any]:
    """knowledge_base.json에서 메타데이터 가져오기 (없으면 빈 dict)."""
    section_data = KNOWLEDGE_BASE.get(section, {})
    return section_data.get(key, {})


def generate_candidates(rule_results: List[Dict[str, Any]],
                        ner_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    # 3-1) 규칙 기반에서 생성된 후보
    for item in rule_results:
        kb_section = item.get("kb_section")
        kb_key = item.get("kb_key")
        if kb_section and kb_key:
            cand_text = kb_key
        else:
            cand_text = item.get("text", "")

        candidate = {
            "text": cand_text,
            "category": item.get("category", ""),
            "source": "rule",
            "start": item.get("start", -1),
            "end": item.get("end", -1),
            "confidence": float(item.get("confidence", 1.0)),
        }

        if kb_section and kb_key:
            candidate["kb_section"] = kb_section
            candidate["kb_key"] = kb_key

        # entity_short_candidates 있으면 각각 확장
        if "entity_short_candidates" in item:
            short_list = item["entity_short_candidates"]
            for short_name in short_list:
                c = candidate.copy()
                c["text"] = short_name
                c["source"] = "rule_pattern_expand"
                candidates.append(c)
        else:
            candidates.append(candidate)

    # 3-2) NER 기반 후보
    for ner_item in ner_results:
        word = ner_item.get("word", "")
        if word.startswith("##"):
            continue
        if re.fullmatch(r"[^\w가-힣]+", word or ""):
            continue

        entity_label = ner_item.get("entity_group") or ner_item.get("entity") or "UNKNOWN"

        candidate = {
            "text": word,
            "category": entity_label,
            "source": "NER",
            "start": ner_item.get("start", -1),
            "end": ner_item.get("end", -1),
            "confidence": float(ner_item.get("score", 0.0)),
        }
        candidates.append(candidate)

    return candidates


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """코사인 유사도 계산 (0 벡터 예외 처리)."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def get_embedding(text: str, max_length: int = 64) -> np.ndarray:
    """
    BERT 기반 문장 임베딩.
    - beomi/KcBERT-base 사용
    - last_hidden_state에 대해 attention mask를 이용한 mean pooling.
    """
    text = text.strip()
    if not text:
        return np.zeros(bert_model.config.hidden_size, dtype=float)

    inputs = bert_tok(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    ).to(bert_device)

    with torch.no_grad():
        outputs = bert_model(**inputs)
        last_hidden = outputs.last_hidden_state  # [1, L, H]
        attn_mask = inputs["attention_mask"]     # [1, L]

        mask = attn_mask.unsqueeze(-1)           # [1, L, 1]
        summed = (last_hidden * mask).sum(dim=1)  # [1, H]
        counts = mask.sum(dim=1)                  # [1, 1]
        counts = torch.clamp(counts, min=1e-9)
        mean_pooled = summed / counts             # [1, H]

    return mean_pooled.squeeze(0).cpu().numpy()


def score_candidates(candidates: List[Dict[str, Any]],
                     context_text: str) -> List[Dict[str, Any]]:
    context_embedding = get_embedding(context_text)

    for candidate in candidates:
        emb_text = candidate.get("text", "")

        kb_section = candidate.get("kb_section")
        kb_key = candidate.get("kb_key")
        if kb_section and kb_key:
            meta = get_kb_metadata(kb_section, kb_key)
            desc = meta.get("description", "")
            keywords = " ".join(meta.get("keywords", []))
            aliases = " ".join(meta.get("aliases", []))
            emb_text = f"{candidate['text']} {desc} {keywords} {aliases}"

        cand_embedding = get_embedding(emb_text)
        sim = cosine_similarity(context_embedding, cand_embedding)
        candidate["similarity"] = sim

    return candidates


def map_final_entity(candidates: List[Dict[str, Any]],
                     threshold: float = 0.5) -> Dict[str, Any] | None:
    """
    최종 엔티티 매핑:
      1) 규칙 기반 후보 우선
      2) 없으면 NER 후보
      3) 둘 다 없으면 None
    """
    # 1) 규칙 기반
    rule_candidates = [c for c in candidates if c.get("source", "").startswith("rule")]
    if rule_candidates:
        sorted_rule = sorted(
            rule_candidates,
            key=lambda x: x.get("similarity", 0.0),
            reverse=True
        )
        best_rule = sorted_rule[0]
        if best_rule.get("similarity", 0.0) >= threshold:
            return best_rule
        return best_rule

    # 2) 규칙 기반이 없을 때 NER 후보
    ner_candidates = [c for c in candidates if c.get("source") == "NER"]
    if ner_candidates:
        sorted_ner = sorted(
            ner_candidates,
            key=lambda x: x.get("similarity", 0.0),
            reverse=True
        )
        best_ner = sorted_ner[0]
        if best_ner.get("similarity", 0.0) >= threshold:
            return best_ner
        return best_ner

    # 3) 후보 없음
    return None

# 3.5단계: KB 기반 문장 재작성 (정규화)

def rewrite_with_kb(text: str,
                    rule_results: List[Dict[str, Any]]) -> str:
    """
    rule_results 중 knowledge_base.json과 연결된 것들을 사용해
    원문 텍스트를 정규화:
      - companies  : full_name 또는 키 이름으로 치환
      - internet_slang : meaning으로 치환
    """
    replacements: List[tuple[int, int, str]] = []

    for item in rule_results:
        kb_section = item.get("kb_section")
        kb_key = item.get("kb_key")
        if not (kb_section and kb_key):
            continue

        meta = KNOWLEDGE_BASE.get(kb_section, {}).get(kb_key, {})
        rep = None

        if kb_section == "companies":
            # full_name이 있으면 사용, 없으면 키(정식 명칭) 사용
            rep = meta.get("full_name", kb_key)
        elif kb_section == "internet_slang":
            # 의미 필드가 있으면 사용 (예: 띵작 → 명작)
            rep = meta.get("meaning", kb_key)
        else:
            # 그 외 섹션은 일단 canonical key로 치환
            rep = kb_key

        if not rep:
            continue

        replacements.append((item["start"], item["end"], rep))

    if not replacements:
        return text

    # 시작 위치 기준 정렬
    replacements.sort(key=lambda x: x[0])

    # 겹치는 스팬은 앞에 나온 것만 사용
    result_parts: List[str] = []
    cur = 0
    for start, end, rep in replacements:
        if start < cur:
            # 이미 처리한 영역과 겹치면 스킵
            continue
        result_parts.append(text[cur:start])
        result_parts.append(rep)
        cur = end
    result_parts.append(text[cur:])

    return "".join(result_parts)


## 전체 파이프라인

def run_pipeline(text: str) -> Dict[str, Any]:
    cleaned_text = preprocess_text(text)

    # 1단계: 규칙 기반 탐지
    rule_results = detect_expressions(cleaned_text)

    # 2단계: KoELECTRA NER
    ner_results = extract_entities(cleaned_text)

    # 3단계: 후보군 생성
    candidates = generate_candidates(rule_results, ner_results)

    # 4단계: BERT 임베딩 기반 코사인 유사도
    candidates_with_scores = score_candidates(candidates, cleaned_text)

    # 5단계: 최종 엔티티 매핑 (요약용)
    final_entity = map_final_entity(candidates_with_scores, threshold=MIN_CONF)

    # 3.5단계: KB 기반 문장 정규화
    normalized_text = rewrite_with_kb(cleaned_text, rule_results)

    return {
        "cleaned_text": cleaned_text,
        "rule_results": rule_results,
        "ner_results": ner_results,
        "candidates": candidates_with_scores,
        "final_entity": final_entity,
        "normalized_text": normalized_text,
    }


# CLI: 한 문장 입력 → 정규화된 문장 출력

if __name__ == "__main__":
    input_text = input("분석할 문장을 입력하세요: ").strip()
    if not input_text:
        print("원문 입력 : ")
        print("분석 결과 : 입력이 비어 있습니다.")
        sys.exit(0)

    result = run_pipeline(input_text)

    print("원문 입력 :", result["cleaned_text"])
    print("분석 결과 :", result["normalized_text"])
    # 디버깅용으로 보고 싶으면 아래도 참고
    # print("최종 엔티티 :", format_final_entity(result["final_entity"]))