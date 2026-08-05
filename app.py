import streamlit as st

from chatbot import generate_rag_response


# --------------------------------------------------
# 1. Streamlit 화면 설정
# --------------------------------------------------

st.set_page_config(
    page_title="English Conversation Tutor",
    page_icon="💬",
)

st.title("English Conversation Tutor")
st.caption("중학생 대상 RAG 영어회화 챗봇 프로토타입")


# --------------------------------------------------
# 2. 레벨 선택
# --------------------------------------------------

selected_level = st.selectbox(
    "영어 레벨을 선택하세요.",
    options=["Level 1", "Level 2", "Level 3"],
)


# --------------------------------------------------
# 3. 세션 상태 초기화
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_level" not in st.session_state:
    st.session_state.selected_level = selected_level

if "last_debug" not in st.session_state:
    st.session_state.last_debug = None


# 레벨이 바뀌면 기존 대화 초기화
if st.session_state.selected_level != selected_level:
    st.session_state.messages = []
    st.session_state.selected_level = selected_level
    st.session_state.last_debug = None
    st.rerun()


# --------------------------------------------------
# 4. 기존 대화 표시
# --------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# 5. 사용자 입력
# --------------------------------------------------

user_text = st.chat_input("영어로 말해보세요.")

if user_text:
    user_message = {
        "role": "user",
        "content": user_text,
    }

    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(user_text)

    try:
        with st.chat_message("assistant"):
            with st.spinner("관련 학습자료를 검색하고 있어요..."):
                answer, debug_info = generate_rag_response(
                    messages=st.session_state.messages,
                    selected_level=selected_level,
                )

            st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        st.session_state.last_debug = debug_info

    except Exception as error:
        st.error(f"응답 생성 중 오류가 발생했습니다: {error}")


# --------------------------------------------------
# 6. 검색 결과 확인
# 개발 중에만 사용하고, 최종 배포 때는 숨겨도 됨
# --------------------------------------------------

if st.session_state.last_debug:
    debug = st.session_state.last_debug

    with st.expander("RAG 검색 결과 확인"):
        st.write("선택 레벨:", debug["selected_level"])
        st.write("검색된 토픽:", debug["retrieved_topic"])
        st.write(
            "토픽 거리 점수:",
            round(debug["topic_score"], 4),
        )

        st.markdown("#### 검색된 레벨 자료")
        st.code(debug["level_material"])

        st.markdown("#### 검색된 토픽 자료")
        st.code(debug["topic_material"])


# --------------------------------------------------
# 7. 초기화 버튼
# --------------------------------------------------

if st.button("대화 초기화"):
    st.session_state.messages = []
    st.session_state.last_debug = None
    st.rerun()