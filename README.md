# dsl-economic-news-chatbot
A chatbot that explains Korean economic news to beginners

A Streamlit-based chatbot that helps Korean economic news beginners understand articles by summarizing, explaining terms, extracting entities, and recommending related news.  
Built using Retrieval-Augmented Generation (RAG), FAISS, GPT-4o, and various NLP modules.

## 🔍 Features

- 경제 초보자도 이해할 수 있는 뉴스 요약 제공
- 뉴스에 등장한 경제 용어 자동 추출 및 RAG 기반 쉬운 설명
- 기업명 추출 및 GPT-기반 기업 설명 생성
- 유사 뉴스 추천 (NER 기반, 사건 기반)
- 경제 퀴즈 자동 생성
- Streamlit으로 간단하게 실행 가능한 인터페이스

---

## ⚙️ Modules Overview

| File | Description |
|------|-------------|
| `chatbot2.py` | Streamlit 기반 메인 챗봇 실행 파일. 전체 UI 및 파이프라인 조율 |
| `news_summarizer.py` | GPT-4o를 사용하여 뉴스 요약 수행 |
| `economic_terms_RAG.py` | GPT 및 BGE-M3 임베딩 기반 경제 용어 설명 및 유사 용어 추천 |
| `company_info.py` | 기업명 NER 결과를 바탕으로 GPT-4o로 기업 설명 생성 |
| `ner.py` | KLUE RoBERTa 기반 개체명 인식(NER) 수행 |
| `ner_similarity_chatbot2.py` | NER 기반 유사 뉴스 추천 (FAISS + SBERT) |
| `news_recommender_final2.py` | 사건 기반 유사 뉴스 추천 (타이틀 + 본문 임베딩, L2 거리 기반 필터링) |
| `generate_quiz_module.py` | 기사 기반 경제 퀴즈 자동 생성 (기업/용어/내용 기반 3문제) |
| `qa_prompt_module.py` | 사용자 질문에 대한 GPT-4o 응답용 프롬프트 생성기 |

---

## 🧰 Tech Stack

- **LLM & NLP**: GPT-4o, Hugging Face Transformers, LangChain
- **Retrieval & Search**: FAISS, Sentence Transformers (BGE-M3, ko-sroberta)
- **Frontend**: Streamlit
- **Data Handling**: Pandas, NumPy
- **Prompt Engineering**: RAG, structured templates, hallucination mitigation

---

## ▶️ How to Run

```bash
# 필요한 라이브러리 설치
pip install -r requirements.txt

# 환경변수 설정
touch .env
# .env 안에 OPENAI_API_KEY=your_key 입력

# 실행
streamlit run chatbot2.py
