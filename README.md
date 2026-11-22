
# Personalized Learning Resource Recommender (AI Lab Project)

A command-line and web-based **hybrid AI recommender system** that suggests courses, books, tutorials, and projects based on a learner's interests, goals, skill level, preferences, and feedback.

Built as an **NLP Lab Project**, this system combines traditional NLP techniques (TF-IDF, cosine similarity, Rocchio feedback, collaborative filtering) with modern Generative AI (LLM-based recommendations, prompt engineering, and a chat assistant) to deliver personalized learning paths.

---

## 🎯 Project Idea

Learners create a profile (skill level, topics of interest, learning goal, preferred formats, time limit, free-only option). The system then:

1. Recommends resources using a **weighted hybrid model** (content-based + collaborative + knowledge-based + popularity + sentiment).
2. Allows **keyword search** over the resource catalogue.
3. Generates an ordered **learning path** from the learner's current level up to Advanced.
4. Shows **similar resources** ("more like this").
5. Lets learners **rate and complete** resources, updating the model immediately (feedback loop).
6. Provides **offline evaluation** (Precision@K, Recall@K, Hit-rate, Coverage vs. a random baseline).
7. Includes an **AI Assistant** (LLM chat) for real-time recommendations, web search, and study-plan guidance.

---

## 📊 Rubric Mapping (Project Demonstration – 30 Marks)

| Sr. No. | Assessment Criteria | CO Mapping | Marks | Project Implementation |
|---------|---------------------|------------|-------|------------------------|
| 1 | Text extraction, preprocessing, cleaning and vectorization of unstructured data | CO1 | 7 | `data_loader.py` loads and cleans CSV data; `engine.py` builds text from title/topic/tags/description and vectorizes with `TfidfVectorizer` (unigrams + bigrams, stop-word removal, sublinear TF). |
| 2 | Implementation and evaluation of sentiment analysis and information extraction using appropriate algorithms | CO2 | 8 | `nlp_features.py` runs **VADER/TextBlob sentiment analysis** on resource descriptions; `evaluation.py` implements hold-out **Precision@K, Recall@K, Hit-rate, Coverage** with a random baseline. |
| 3 | Implementation of topic modeling or multilingual NLP using appropriate NLP techniques and cloud-based AI services | CO3 | 7 | `nlp_features.py` implements **LDA/NMF topic modeling** to discover hidden topics from descriptions; **MultilingualSupport** detects query language (`langdetect`) and translates non-English queries to English (`deep-translator`). Cloud AI via OpenAI GPT-4o-mini API. |
| 4 | Generative AI/LLM application, prompt engineering, integration and working demonstration | CO4 | 8 | `groq_client.py` uses **LLM (gpt-4o-mini)** for real-time recommendations, explanations, web search, and a multi-turn chat assistant. Integrated into Streamlit UI with custom prompts and JSON output parsing. |
| **Total** | **CO1–CO4** | | **30** | |

---

## 🚀 Features

1. **Learner Profiles** — skill level, interests, goal, preferred formats, time limit, free-only option. Saved as JSON in `data/users/`.
2. **Personalized Recommendations** with explanations ("Why recommended?") and score breakdown per signal.
3. **Keyword Search** ranked by TF-IDF relevance.
4. **Learning Path Generator** — ordered path from learner's level up to Advanced (Beginner → Intermediate → Advanced).
5. **Similar Resources** — "more like this" using content + collaborative similarity.
6. **Feedback Loop** — rating or completing a resource updates the collaborative model immediately.
7. **Offline Evaluation** — Precision@K, Recall@K, Hit-rate, Coverage compared against random baseline.
8. **AI Assistant** — LLM-powered chat for study advice, real-time picks, and web-searched resources.
9. **Sentiment Analysis** — VADER/TextBlob scoring of resource descriptions (boosts positively-written content).
10. **Topic Modeling** — LDA/NMF discovers hidden themes from resource descriptions.
11. **Multilingual Support** — detects non-English search queries and translates them to English before searching.

---

## 🛠️ NLP Techniques Used

| Component | Technique | Library |
|-----------|-----------|---------|
| Content-based filtering | TF-IDF vectorisation (unigrams + bigrams) + cosine similarity | scikit-learn |
| Relevance feedback | Rocchio algorithm: profile = interests + liked items − disliked items | numpy |
| Collaborative filtering | Item-item CF using adjusted-cosine similarity with shrinkage | pandas, scikit-learn |
| Knowledge-based filtering | Rules for skill-level fit and preferred format; hard constraints for budget/duration | pandas |
| Hybridisation | Weighted hybrid of all signals; weights re-normalised during cold start | numpy |
| Explainability | Rule-generated reasons for every recommendation | — |
| Sentiment analysis | VADER compound polarity (fallback: TextBlob) | vaderSentiment, textblob |
| Topic modeling | LDA / NMF on resource descriptions | scikit-learn |
| Multilingual NLP | Language detection + translation to English | langdetect, deep-translator |
| Generative AI | Prompt engineering + LLM chat/completion | OpenAI SDK (gpt-4o-mini) |

---

## 📁 Project Structure

```text
NLP-project-/
│
├── main.py                     # CLI entry point
├── streamlit_app.py            # Streamlit web UI entry point
├── requirements.txt             # Python dependencies
├── .env                         # API keys (NOT committed)
├── .gitignore                   # Git ignore rules
├── README.md                    # Project documentation
│
├── data/
│   ├── resources.csv            # 58 learning resources (catalogue)
│   ├── ratings.csv              # Community ratings (synthetic)
│   └── users/                   # Saved learner profiles (JSON)
│
├── recommender/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── profile.py
│   ├── engine.py
│   ├── evaluation.py
│   ├── groq_client.py
│   └── nlp_features.py
│
├── cli/
│   ├── app.py
│   └── display.py
│
└── scripts/
    └── generate_ratings.py
```

---

## ⚡ Installation & Running

### 1. Clone the Repository

```bash
git clone https://github.com/AmrutaKanse09/NLP_Project.git
cd NLP-project-
```

### 2. Create and Activate a Virtual Environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Mac/Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** `requirements.txt` includes streamlit, pandas, scikit-learn, numpy, openai, python-dotenv, vaderSentiment, textblob, langdetect, and deep-translator.

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_CHAT_MODEL=gpt-4o-mini
```

Get an API key from [OpenAI Platform](https://platform.openai.com/).

### 5. Download TextBlob Corpora

```bash
python -m textblob.download_corpora
```

### 6. Run the Project

**Option A: Command Line Interface (CLI)**

```bash
python main.py
```

**Option B: Streamlit Web UI (Recommended)**

```bash
streamlit run streamlit_app.py
```

This opens `http://localhost:8501` in your browser.

The UI has tabs for:

- Recommendations (with ⭐ rating)
- Search (supports multilingual queries)
- Learning Path
- Browse & Similar
- My Profile
- Evaluation (with chart)
- 🤖 AI Assistant
