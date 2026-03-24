# Intelligent Resume Agent System

An AI-powered resume management system with **autonomous agent capabilities** using the ReAct (Reasoning + Acting) pattern. The system can intelligently search, filter, and analyze resumes through multi-step reasoning.

## Features

### Core Capabilities
- 📄 **PDF Resume Processing** - Extract and process resume content
- 🔍 **Semantic Search** - Find candidates using natural language
- 🎯 **Metadata Extraction** - Automatically extract structured information
- 💾 **Vector Database** - ChromaDB for efficient similarity search
- 🤖 **RAG System** - Retrieval-Augmented Generation for Q&A

### 🆕 Intelligent Agent (Phase 1)
- 🧠 **Multi-Step Reasoning** - Breaks down complex queries autonomously
- 🛠️ **Tool Selection** - Automatically chooses appropriate tools
- 🔄 **ReAct Pattern** - Thought → Action → Observation loop
- 📊 **Reasoning Trace** - Transparent decision-making process
- 🎯 **Goal-Driven** - Iterates until query is fully answered

### Available Agent Tools
1. **search_resumes** - Semantic search in resume database
2. **filter_by_metadata** - Filter by skills, experience, location, etc.
3. **summarize_resume** - Generate detailed candidate summaries

## Installation

1. **Clone or navigate to the project directory**

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Add Resumes

Place your PDF resumes in the `resumes/` folder (created automatically on first run).
There are 10 mock resumes already placed in that folder as sample.

### 2. Load Resumes into Vector Database

```bash
python main.py --load
```

This will:
- Extract text from all PDF files in the `resumes/` folder
- Chunk the text into manageable pieces
- Store embeddings in ChromaDB

### 3. Query Resumes

```bash
python main.py --query
```

This starts an interactive chat session where you can ask questions like:
- "What candidates have Python experience?"
- "Who has worked with machine learning?"
- "Find candidates with 5+ years of experience"
- "Which resumes mention AWS or cloud computing?"

### 4. Load and Query (Combined)

```bash
python main.py --load-and-query
```

This loads resumes and immediately starts the query interface.

### 5. Add Resume (Basic)
```bash
python main.py --add-resume <path_to_pdf>
```
This adds single resume files without reloading the entire database while automatically detecting duplicates based on metadata (name, email, phone)

### 6. 🆕 Intelligent Agent Mode (Recommended)
```bash
python main.py --agent
```
This starts the **autonomous agent** that can:
- Understand complex natural language queries
- Break down tasks into multiple steps
- Select and use appropriate tools automatically
- Provide comprehensive answers with reasoning trace

**Example queries:**
- "Find Python developers with AWS experience in Bangalore"
- "Who has the most machine learning experience?"
- "Show me senior backend engineers and summarize the top candidate"

See [AGENT_GUIDE.md](AGENT_GUIDE.md) for detailed documentation.

## Agent vs Traditional RAG

| Mode | Best For | Complexity |
|------|----------|------------|
| `--query` | Simple questions, single-step queries | Low |
| `--agent` | Complex queries, multi-criteria search, analysis | High |

**Agent Mode Benefits:**
- ✅ Handles complex multi-step queries
- ✅ Autonomous tool selection
- ✅ Transparent reasoning process
- ✅ Better context understanding
- ✅ More comprehensive answers

## Quick Start Examples

### Traditional RAG Mode
```bash
# Load resumes
python main.py --load

# Query mode
python main.py --query
> What candidates have Python experience?
```

### Agent Mode (Recommended)
```bash
# Load resumes
python main.py --load

# Agent mode
python main.py --agent
> Find senior Python developers in Mumbai with 5+ years experience and show me their details
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface (CLI)                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Agent Controller                          │
│              (ReAct: Reasoning + Acting)                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Thought → Action → Observation → Repeat             │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Tool Registry                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Search     │  │   Filter     │  │  Summarize   │     │
│  │   Resumes    │  │  Metadata    │  │   Resume     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Vector Store (ChromaDB)                   │
│              + Metadata Extraction (LLM)                     │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
Intelligent_CMS/
├── tools/                      # 🆕 Agent tools
│   ├── __init__.py            # Tool registry
│   ├── base_tool.py           # Abstract base class
│   ├── search_tool.py         # Semantic search
│   ├── filter_tool.py         # Metadata filtering
│   └── summarize_tool.py      # Resume summarization
├── agent_controller.py         # 🆕 ReAct agent logic
├── main.py                     # CLI interface (updated)
├── config.py                   # Configuration (updated)
├── pdf_processor.py            # PDF text extraction
├── vector_store.py             # ChromaDB interface
├── rag_system.py               # RAG implementation
├── metadata_extractor.py       # LLM-based extraction
├── resumes/                    # Resume PDFs
├── AGENT_GUIDE.md             # 🆕 Agent documentation
└── README.md                   # This file
```

## Configuration

Edit `config.py` to customize:

```python
# Agent settings
AGENT_MODEL = "llama3.2:latest"      # LLM for reasoning
AGENT_MAX_ITERATIONS = 5              # Max reasoning steps
AGENT_TEMPERATURE = 0.3               # Reasoning focus
ENABLE_REASONING_TRACE = True         # Show thinking

# Vector DB settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RESULTS = 3
```

## Requirements

- Python 3.8+
- Ollama with llama3.2:latest model
- ChromaDB
- See `requirements.txt` for full list

## Documentation

- **[AGENT_GUIDE.md](AGENT_GUIDE.md)** - Complete agent system documentation
- **[README.md](README.md)** - This file (overview and setup)

## Roadmap

### ✅ Phase 1 (Current)
- Core agent with ReAct pattern
- 3 essential tools (search, filter, summarize)
- Multi-step reasoning
- Reasoning trace visibility

### 🔄 Phase 2 (Planned)
- Conversation memory across sessions
- Additional tools (compare, rank, analyze)
- Self-correction capabilities
- Performance optimization

### 🔮 Phase 3 (Future)
- Multi-agent collaboration
- Learning from feedback
- Proactive suggestions
- Advanced analytics