# European Cross-Commodity Risk Monitor

## Setup

```bash
pip install -r requirements.txt
```

### Ollama + Qwen (required for AI narrative)
1. Install Ollama: https://ollama.com/download
2. Pull the model:
```bash
ollama pull qwen2.5:3b
```

## Run

```bash
python ai_llm.py
```

This pulls metrics automatically and generates the AI desk note. Output is written to `out/`. LLM prompts and responses are logged in `out/logs/`.