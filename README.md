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