# AI Examples

![Python](https://img.shields.io/badge/python-3.8%2B-green)
![Last Updated](https://img.shields.io/badge/last%20updated-April%202025-blue)
[![Python Tests](https://github.com/garrigueta/ai-examples/actions/workflows/python-tests.yml/badge.svg)](https://github.com/garrigueta/ai-examples/actions/workflows/python-tests.yml)

A collection of AI implementation examples and tools for various use cases, from flight simulation assistants to local LLM interactions.

## Overview

This repository contains practical examples of AI integration in different contexts, demonstrating how to use various AI technologies, including:

- Voice assistants using OpenAI GPT and VOSK
- Local LLM interactions with Ollama models
- Document retrieval systems using embedding and vector search
- AI-powered terminal command generation and interpretation
- Semantic search implementations
- Flight Simulator (MSFS) AI assistant

## Examples Included

### FlightGPT (MSFS Assistant)
A voice-controlled AI assistant for Microsoft Flight Simulator that provides real-time flight data through natural language conversations.

### Local LLM Interactions
Tools for interacting with locally hosted language models via Ollama:
- Command generation from natural language descriptions
- Error explanation and troubleshooting
- Prompt-based interactions with local models

### RAG (Retrieval Augmented Generation)
Examples of document retrieval systems that:
- Load documents from directories
- Create embeddings and vector stores
- Support history-aware retrieval chains
- Implement question-answering capabilities

### Semantic Search
Implementation of semantic search using embeddings to find relevant content based on meaning rather than exact match.

## Requirements

Core dependencies:
- Python 3.8+
- OpenAI API (for some examples)
- Ollama (for local LLM examples)
- VOSK (for speech recognition)
- Additional libraries in `reqs.txt`

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ai-examples.git
   cd ai-examples
   ```

2. Install required packages:
   ```bash
   pip install -r reqs.txt
   ```

3. For speech recognition, download a VOSK model from https://alphacephei.com/vosk/models and place it in the `vosk` folder

4. For examples using OpenAI, set up your API key:
   ```bash
   export OPENAI_API_KEY=your_api_key
   export OPENAI_ORG=your_organization_id
   export OPENAI_PROJ=your_project_id
   ```

5. For local LLM examples, install and set up Ollama:
   ```bash
   # For Linux
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Start Ollama service
   ollama serve
   
   # In a new terminal, pull a model (e.g., gemma3, llama3, mistral)
   ollama pull gemma3
   ```
   
   For more details, visit the [Ollama installation guide](https://github.com/ollama/ollama#installation).

## Usage Examples

### FlightGPT (MSFS Assistant)
```bash
python msfs_assitant.py
```

### Semantic Search
```bash
python semantic_search.py
```

### Local LLM Command Generation
```bash
python home/bin/ollama_prompt.py run "list all Python files in the current directory"
```

### Local LLM Prompt
```bash
python home/bin/ollama_prompt.py prompt "Explain how vector embeddings work"
```

### Local LLM Error Explanation
```bash
python home/bin/ollama_prompt.py error "pip install tensorflow" "ERROR: Could not find a version that satisfies the requirement tensorflow"
```

## Project Structure

- `msfs_assitant.py`: FlightGPT application entry point
- `semantic_search.py`: Example of semantic search implementation
- `lib/`: Core libraries and modules
  - `modules/`: Basic functionality modules
    - `ai.py`: AI interaction wrapper for OpenAI
    - `audio.py`: Voice recognition with VOSK
    - `msfs.py`: Microsoft Flight Simulator interface
    - `speech.py`: Text-to-speech utilities
  - `mcp/`: Model Context Protocol implementation
    - `actions.py`: Terminal command generation and execution
    - `transport.py`: MCP message handling
  - `chat/`: Chat implementations
    - `openai.py`: OpenAI-based chat
    - `history_ollama.py`: Chat with history using Ollama
  - `storage/`: Data storage and retrieval
    - `docs.py`: Document storage and RAG implementation
  - `backend/`: Backend services
    - `run.py`: Command execution API
- `home/bin/`: Utility scripts
  - `ollama_prompt.py`: CLI for interacting with Ollama

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
