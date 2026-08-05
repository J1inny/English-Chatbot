import os
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
import streamlit as st


# --------------------------------------------------
# 1. 경로와 API 설정
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

LEVEL_DB_PATH = BASE_DIR / "chroma_db" / "levels"
TOPIC_DB_PATH = BASE_DIR / "chroma_db" / "topics"

load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    api_key = st.secrets.get("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY를 찾지 못했습니다.")

client = OpenAI(api_key=api_key)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,
)


# --------------------------------------------------
# 2. 기존 ChromaDB 불러오기
# --------------------------------------------------

if not LEVEL_DB_PATH.exists():
    raise FileNotFoundError(
        "레벨 DB가 없습니다. 먼저 python build_db.py를 실행하세요."
    )

if not TOPIC_DB_PATH.exists():
    raise FileNotFoundError(
        "토픽 DB가 없습니다. 먼저 python build_db.py를 실행하세요."
    )

level_db = Chroma(
    collection_name="level_rules",
    embedding_function=embeddings,
    persist_directory=str(LEVEL_DB_PATH),
)

topic_db = Chroma(
    collection_name="topics",
    embedding_function=embeddings,
    persist_directory=str(TOPIC_DB_PATH),
)


# --------------------------------------------------
# 3. 짧은 시스템 프롬프트
# --------------------------------------------------

SYSTEM_PROMPT = """
You are an English conversation tutor for Korean middle school students.

Rules

- Speak mainly in English.
- Match the student's level.
- Keep responses short.
- Ask one follow-up question.
- Correct only important mistakes.
- Continue the conversation naturally.
- When the student ends the conversation, provide:
  1. Korean feedback
  2. Better English expressions
  3. 2 review questions.

Safety

- Refuse sexual roleplay.
- Refuse profanity generation.
- Refuse dangerous or illegal requests.
- Briefly explain rude words only when educational.
"""


# --------------------------------------------------
# 4. 선택한 레벨 자료 검색
# --------------------------------------------------

def retrieve_level_material(selected_level: str) -> str:
    """
    'Level 1' 같은 화면 선택값을 숫자 1로 변환한 뒤
    해당 레벨 자료를 메타데이터 필터로 검색한다.
    """

    try:
        level_number = int(selected_level.split()[-1])
    except (ValueError, IndexError) as error:
        raise ValueError(
            f"올바르지 않은 레벨 형식입니다: {selected_level}"
        ) from error

    results = level_db.similarity_search(
        query=f"English learning rules for Level {level_number}",
        k=1,
        filter={"level": level_number},
    )

    if not results:
        raise ValueError(
            f"Level {level_number} 자료를 DB에서 찾지 못했습니다."
        )

    return results[0].page_content


# --------------------------------------------------
# 5. 사용자 발화와 관련된 토픽 검색
# --------------------------------------------------

def retrieve_topic_material(
    user_text: str,
) -> Tuple[str, str, float]:
    """
    사용자 입력과 의미가 가장 가까운 토픽 한 개를 검색한다.

    반환값:
    - 검색된 문서 내용
    - 검색된 토픽 이름
    - 거리 점수
    """

    results = topic_db.similarity_search_with_score(
        query=user_text,
        k=1,
    )

    if not results:
        return "", "검색 결과 없음", float("inf")

    document, score = results[0]

    topic_name = document.metadata.get(
        "topic",
        "알 수 없는 토픽",
    )

    return document.page_content, topic_name, float(score)


# --------------------------------------------------
# 6. GPT에 전달할 RAG 지침 구성
# --------------------------------------------------

def build_instructions(
    level_material: str,
    topic_material: str,
) -> str:
    return f"""
{SYSTEM_PROMPT}

Retrieved level material:
--- LEVEL START ---
{level_material}
--- LEVEL END ---

Retrieved topic material:
--- TOPIC START ---
{topic_material}
--- TOPIC END ---
""".strip()


# --------------------------------------------------
# 7. RAG 기반 GPT 응답 생성
# --------------------------------------------------

def generate_rag_response(
    messages: List[Dict[str, str]],
    selected_level: str,
) -> Tuple[str, Dict[str, object]]:
    """
    대화 기록의 마지막 사용자 입력으로 토픽을 검색하고,
    선택한 레벨 자료와 함께 GPT에 전달한다.

    반환값:
    - GPT 답변
    - 검색 확인용 디버그 정보
    """

    if not messages:
        raise ValueError("대화 기록이 비어 있습니다.")

    user_messages = [
        message
        for message in messages
        if message.get("role") == "user"
    ]

    if not user_messages:
        raise ValueError("사용자 입력을 찾을 수 없습니다.")

    latest_user_text = user_messages[-1]["content"].strip()

    if not latest_user_text:
        raise ValueError("마지막 사용자 입력이 비어 있습니다.")

    level_material = retrieve_level_material(selected_level)

    (
        topic_material,
        retrieved_topic,
        topic_score,
    ) = retrieve_topic_material(latest_user_text)

    instructions = build_instructions(
        level_material=level_material,
        topic_material=topic_material,
    )

    # 개발 중 검색 결과 확인
    print("\n" + "=" * 70)
    print("선택 레벨:", selected_level)
    print("사용자 입력:", latest_user_text)
    print("검색 토픽:", retrieved_topic)
    print("토픽 거리 점수:", topic_score)
    print("=" * 70)

    response = client.responses.create(
        model="gpt-4.1-mini",
        instructions=instructions,
        input=messages,
    )

    debug_info = {
        "selected_level": selected_level,
        "retrieved_topic": retrieved_topic,
        "topic_score": topic_score,
        "level_material": level_material,
        "topic_material": topic_material,
    }

    return response.output_text, debug_info
