import json
import os
import shutil
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


# --------------------------------------------------
# 1. 경로 및 환경변수
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

LEVEL_JSON_PATH = DATA_DIR / "level_rules.json"
TOPIC_JSON_PATH = DATA_DIR / "topics.json"

LEVEL_DB_PATH = BASE_DIR / "chroma_db" / "levels"
TOPIC_DB_PATH = BASE_DIR / "chroma_db" / "topics"

load_dotenv(BASE_DIR / ".env")

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY를 찾지 못했습니다. "
        ".env 파일이 저장되었는지 확인하세요."
    )


# --------------------------------------------------
# 2. JSON 읽기
# --------------------------------------------------

def load_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾지 못했습니다: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"{path.name}의 최상위 구조는 리스트여야 합니다.")

    return data


# --------------------------------------------------
# 3. 레벨 데이터를 LangChain Document로 변환
# --------------------------------------------------

def create_level_documents(
    level_data: list[dict[str, Any]],
) -> list[Document]:
    documents: list[Document] = []

    for item in level_data:
        grammar_text = "\n".join(
            f"- {grammar}" for grammar in item.get("grammar", [])
        )
        vocabulary_text = "\n".join(
            f"- {word}" for word in item.get("vocabulary", [])
        )

        # 실제 임베딩되는 텍스트
        page_content = f"""
English learner level: Level {item["level"]}

Recommended grammar:
{grammar_text}

Recommended vocabulary and language areas:
{vocabulary_text}
""".strip()

        document = Document(
            page_content=page_content,
            metadata={
                "id": item["id"],
                "data_type": item["data_type"],
                "level": item["level"],
            },
        )

        documents.append(document)

    return documents


# --------------------------------------------------
# 4. 토픽 데이터를 LangChain Document로 변환
# --------------------------------------------------

def create_topic_documents(
    topic_data: list[dict[str, Any]],
) -> list[Document]:
    documents: list[Document] = []

    for item in topic_data:
        expressions_text = "\n".join(
            f"- {expression}"
            for expression in item.get("key_expressions", [])
        )
        questions_text = "\n".join(
            f"- {question}"
            for question in item.get("assistant_questions", [])
        )

        # 사용자 입력과 의미 유사도를 비교할 핵심 텍스트
        page_content = f"""
Topic: {item["topic"]}

Description:
{item["topic_description"]}

Useful expressions:
{expressions_text}

Possible follow-up questions:
{questions_text}
""".strip()

        document = Document(
            page_content=page_content,
            metadata={
                "id": item["id"],
                "data_type": item["data_type"],
                "topic": item["topic"],
            },
        )

        documents.append(document)

    return documents


# --------------------------------------------------
# 5. 기존 DB 삭제
# 같은 문서를 중복 저장하지 않기 위해 매번 새로 생성
# --------------------------------------------------

def remove_existing_db(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
        print(f"기존 DB 삭제: {path}")


# --------------------------------------------------
# 6. ChromaDB 생성
# --------------------------------------------------

def build_vector_databases() -> None:
    level_data = load_json(LEVEL_JSON_PATH)
    topic_data = load_json(TOPIC_JSON_PATH)

    level_documents = create_level_documents(level_data)
    topic_documents = create_topic_documents(topic_data)

    print(f"레벨 문서 수: {len(level_documents)}")
    print(f"토픽 문서 수: {len(topic_documents)}")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    remove_existing_db(LEVEL_DB_PATH)
    remove_existing_db(TOPIC_DB_PATH)

    Chroma.from_documents(
        documents=level_documents,
        embedding=embeddings,
        ids=[doc.metadata["id"] for doc in level_documents],
        collection_name="level_rules",
        persist_directory=str(LEVEL_DB_PATH),
    )

    print(f"레벨 DB 생성 완료: {LEVEL_DB_PATH}")

    Chroma.from_documents(
        documents=topic_documents,
        embedding=embeddings,
        ids=[doc.metadata["id"] for doc in topic_documents],
        collection_name="topics",
        persist_directory=str(TOPIC_DB_PATH),
    )

    print(f"토픽 DB 생성 완료: {TOPIC_DB_PATH}")


if __name__ == "__main__":
    build_vector_databases()