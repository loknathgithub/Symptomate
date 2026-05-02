<div align="center">
  <h1>🩺 Symptomate</h1>
  <p>An Intelligent Medical Chatbot powered by a Generator-Validator LLM Architecture</p>
  
  [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
  [![LangChain](https://img.shields.io/badge/LangChain-Enabled-lightgrey.svg)](https://python.langchain.com/)
  [![Chainlit](https://img.shields.io/badge/Chainlit-UI-coral.svg)](https://docs.chainlit.io/)
  [![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-orange.svg)](https://groq.com/)
</div>

<br />

## 📖 Overview

**Symptomate** is a state-of-the-art medical chatbot designed to help users understand their symptoms. Built as a comprehensive final year project, it employs a sophisticated **Generator-Validator architecture** to ensure the highest quality and accuracy of responses. By retrieving medical context using a BM25 index and refining answers through a multi-stage validation pipeline, Symptomate provides reliable, context-aware assistance.

> **Disclaimer**: Symptomate is an informational project and should not be used as a substitute for professional medical advice, diagnosis, or treatment.

---

## ✨ Key Features

- **Generator-Validator Pipeline**: Employs two distinct LLM roles (Generator and Validator) to generate an answer, critique it, and regenerate a highly accurate final response.
- **Advanced RAG Architecture**: Utilizes `Rank-BM25` for efficient document retrieval, ensuring context is highly relevant to the user's queries.
- **Hugging Face Integration**: Automatically pulls pre-built vector indices from the Hugging Face Hub, or builds and uploads them dynamically from local PDFs.
- **Modern UI**: Features a clean, asynchronous chat interface powered by `Chainlit`.
- **High Performance**: Leverages the blistering speed of the Groq API and `llama-3.3-70b-versatile` models.

---

## 🛠 Tech Stack

- **Backend / Logic**: Python, LangChain
- **UI Framework**: Chainlit
- **LLM Provider**: Groq API (`llama-3.3-70b-versatile`)
- **Retrieval Engine**: BM25 Retriever, PyPDF
- **Model / Data Hosting**: Hugging Face Hub
- **Package Management**: `uv`

---

## ⚙️ Architecture

Symptomate's response pipeline runs in a 2-round generation process:
1. **Retrieval**: User question is passed to a BM25 retriever to extract relevant context from medical documents.
2. **Round 1 (Generation & Validation)**: The Generator LLM creates an initial answer. The Validator LLM reviews this answer against the context and provides a critique.
3. **Round 2 (Regeneration & Final Validation)**: The Generator LLM takes the initial answer, context, and critique to formulate an improved final response, which is then streamed to the user.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) (Extremely fast Python package installer and resolver)
- API Keys: [Groq API Key](https://console.groq.com/keys) and [Hugging Face Token](https://huggingface.co/settings/tokens)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Symptomate
   ```

2. **Create a virtual environment and activate it**
   ```bash
   uv venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   # source .venv/bin/activate
   ```

3. **Install the dependencies**
   ```bash
   uv pip install -e .
   ```

### Environment Configuration

Create a `.env` file in the root directory of the project and add your API credentials:

```ini
# Groq API Keys (used for Generator and Validator roles)
GROQ_API_KEY1=your_groq_api_key_1
GROQ_API_KEY2=your_groq_api_key_2
GROQ_BASE_URL=https://api.groq.com/openai/v1

# Hugging Face Configuration (For BM25 Index caching)
HF_TOKEN=your_hugging_face_token
```

### Running the Application

Start the Chainlit server by running:

```bash
chainlit run app.py -w
```

The application will automatically open in your default browser at `http://localhost:8000`.

---

## 📁 Project Structure

```text
Symptomate/
├── data/               # Medical PDF documents for building the index
├── src/
│   ├── pipeline.py     # Generator-Validator pipeline logic
│   ├── prompts.py      # System prompts for generation, validation, and critique
│   └── retriever.py    # BM25 retrieval and HF Hub integration
├── app.py              # Main Chainlit application entry point
├── pyproject.toml      # Project dependencies and configurations
├── .env                # Environment variables (not tracked in git)
└── README.md           # Project documentation
```

---

## 📜 License

This project is licensed under the MIT License.
