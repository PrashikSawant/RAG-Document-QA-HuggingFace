# 🚀 RAG Document Q&A — Live on HuggingFace

> My first ever deployed AI application — live on the internet for anyone to use.
> Built on Day 16 of my 4-month journey to become a job-ready AI Engineer.

---

## 🌱 Why I Built This

Building AI apps locally is one thing.
Deploying them so the world can use them is another.

On Day 16, I took my RAG system from Day 14 and made it live.
This was my first real deployment — and it taught me more than
any tutorial ever could.

---

## 💭 Thought Process

I had a working RAG pipeline locally.
The challenge was: how do I make it accessible to everyone?

The answer involved:
- Docker containers (packaging the app)
- HuggingFace Spaces (hosting the app)
- Git LFS issues (learning the hard way)
- Dependency conflicts (debugging in production)

Every error was a lesson. Every fix made me a better engineer.

---

## 🛠️ What This Project Does

A live RAG-powered Document Q&A system that:
- Auto-loads the "Attention Is All You Need" research paper
- Answers questions based on the document content
- Cites which document the answer came from
- Runs 24/7 on HuggingFace Spaces — free

---

## ⚙️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.12 | Core language |
| Streamlit | Web interface |
| Groq API (LLaMA 3.3 70B) | AI responses |
| Sentence Transformers | Text embeddings |
| NumPy | Vector similarity search |
| PyPDF | PDF text extraction |
| Docker | App containerization |
| HuggingFace Spaces | Cloud deployment |
| Git | Version control |

---

## 🚀 How to Run Locally

1. Clone the repository
