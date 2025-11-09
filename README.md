# PDF Chatbot with Ollama - Complete Setup Guide

A Streamlit-based PDF chatbot application that uses Ollama for local AI processing. Upload PDF documents and ask questions about their content using local LLM models - no API keys required!

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Installation Overview](#installation-overview)
- [Terminal 1: Ollama Setup](#terminal-1-ollama-setup)
- [Terminal 2: Application Setup](#terminal-2-application-setup)
- [Running the Application](#running-the-application)
- [How to Use](#how-to-use)
- [Features](#features)
- [Troubleshooting](#troubleshooting)
- [Technical Details](#technical-details)

---

## Prerequisites

Before you begin, ensure you have:

- **macOS** (this guide is for macOS, but can be adapted for Linux/Windows)
- **Python 3.13** or higher installed
- **Homebrew** installed (for installing Ollama)
- **Terminal** access
- **At least 8GB RAM** (for running Ollama models)
- **Internet connection** (for initial model downloads)

---

## Installation Overview

This application requires **two separate terminals** running simultaneously:

1. **Terminal 1**: Runs Ollama server (must stay running)
2. **Terminal 2**: Runs the Streamlit application

Both terminals need to be open and running for the application to work.

---

## Terminal 1: Ollama Setup

### Step 1: Install Ollama

Open **Terminal 1** and install Ollama using Homebrew:

```bash
# Install Ollama
brew install ollama
```

**Alternative installation method:**
If Homebrew is not available, download Ollama directly:
- Visit: https://ollama.ai
- Download the macOS installer
- Follow the installation wizard

### Step 2: Verify Ollama Installation

Check if Ollama is installed correctly:

```bash
# Check Ollama version
ollama --version
```

You should see output like: `ollama version is 0.12.10` (or similar)

### Step 3: Start Ollama Server

Start the Ollama server (this must keep running):

```bash
# Start Ollama server
ollama serve
```

**Expected output:**
```
INFO[0000] Starting server... 
INFO[0000] Listening on 127.0.0.1:11434
```

**⚠️ IMPORTANT:** 
- **Keep this terminal window open** - Ollama must keep running
- **Do NOT close this terminal** - The server needs to stay active
- You'll see logs here when the application uses Ollama

### Step 4: Download Required Models

In a **new terminal window** (or open another terminal), download the required models:

```bash
# Download the embedding model (required for PDF processing)
ollama pull nomic-embed-text

# Download the chat model (required for answering questions)
ollama pull llama2
```

**Expected output:**
```
pulling manifest 
pulling 274302450 bytes ████████████████████ 100% 
pulling 274302450 bytes ████████████████████ 100% 
verifying sha256 digest 
writing manifest 
success
```

**Note:** 
- `nomic-embed-text` is ~274MB (downloads quickly)
- `llama2` is ~3.8GB (may take several minutes depending on your internet speed)

### Step 5: Verify Models are Installed

Check that both models are available:

```bash
# List all installed models
ollama list
```

**Expected output:**
```
NAME                   ID              SIZE      MODIFIED
llama2:latest          78e26419b446    3.8 GB    2 hours ago
nomic-embed-text:latest 0a109f422b47    274 MB    2 hours ago
```

### Step 6: Test Ollama Connection

Verify that Ollama is running and accessible:

```bash
# Test the API connection
curl http://localhost:11434/api/tags
```

**Expected output:** JSON response with model information

**Alternative test:**
```bash
# Test embedding generation
ollama run nomic-embed-text "test"
```

You should see a long array of numbers (the embedding vector).

---

## Terminal 2: Application Setup

### Step 1: Navigate to Project Directory

Open **Terminal 2** (a new terminal window) and navigate to your project:

```bash
# Navigate to your project directory
cd /Users/rithvik/projects/gen-ai-pdf-insights
```

**Verify you're in the right directory:**
```bash
# List files to confirm
ls -la
```

You should see:
- `Chatbot-Code.py`
- `README.md`
- `requirements.txt` (if it exists)
- `gen_ai_env/` (virtual environment folder)

### Step 2: Create Virtual Environment (if not already created)

Create a Python virtual environment:

```bash
# Create virtual environment
python3 -m venv gen_ai_env
```

**Expected output:** (no errors, just returns to prompt)

### Step 3: Activate Virtual Environment

Activate the virtual environment:

```bash
# Activate virtual environment
source gen_ai_env/bin/activate
```

**Expected output:** Your prompt should now show `(gen_ai_env)` at the beginning:
```
(gen_ai_env) rithvik@Rithviks-MacBook-Pro gen-ai-pdf-insights %
```

**⚠️ IMPORTANT:** 
- You must activate the virtual environment every time you open a new terminal
- The `(gen_ai_env)` prefix confirms it's activated

### Step 4: Install Python Dependencies

Install all required Python packages:

```bash
# Install all dependencies at once
pip install streamlit PyPDF2 langchain-ollama langchain langchain-community langchain-classic langchain-text-splitters faiss-cpu requests
```

**Expected output:**
```
Collecting streamlit
  Downloading streamlit-1.51.0-py3-none-any.whl (10.2 MB)
...
Successfully installed streamlit-1.51.0 ...
```

**Note:** This may take 2-5 minutes depending on your internet speed.

### Step 5: Verify Installation

Check that all packages are installed:

```bash
# Verify key packages
pip list | grep -E "streamlit|langchain|faiss|PyPDF2"
```

**Expected output:**
```
faiss-cpu                   1.12.0
langchain                   1.0.5
langchain-classic           1.0.0
langchain-community         0.4.1
langchain-ollama            1.0.0
langchain-text-splitters    1.0.0
PyPDF2                      3.0.1
streamlit                   1.51.0
```

### Step 6: Verify Ollama is Running

Before starting the app, ensure Ollama is running in Terminal 1:

```bash
# Test Ollama connection from Terminal 2
curl http://localhost:11434/api/tags
```

**Expected output:** JSON response with your models

**If you get an error:**
- Go back to **Terminal 1** and make sure `ollama serve` is running
- If not, start it: `ollama serve`

---

## Running the Application

### Step 1: Ensure Both Terminals are Ready

**Terminal 1:**
- ✅ Ollama server is running (`ollama serve`)
- ✅ You see: `Listening on 127.0.0.1:11434`

**Terminal 2:**
- ✅ Virtual environment is activated (`(gen_ai_env)` in prompt)
- ✅ You're in the project directory
- ✅ All dependencies are installed

### Step 2: Start the Streamlit Application

In **Terminal 2**, run:

```bash
# Start the Streamlit app
streamlit run Chatbot-Code.py
```

<<<<<<< HEAD
**Expected output:**
```
You can now view your Streamlit app in your browser.
=======
<img width="1481" alt="image" src="https://github.com/user-attachments/assets/a2d1ff35-1a5d-44ea-9675-64b035a9514e" />

>>>>>>> ad6f630cd65a9376eceb98b7712b7f5db4074c07

  Local URL: http://localhost:8501
  Network URL: http://192.168.29.196:8501

For better performance, install the Watchdog module:
  $ xcode-select --install
  $ pip install watchdog
```

### Step 3: Open the Application

The application will automatically open in your default web browser. If it doesn't:

1. Open your web browser
2. Navigate to: `http://localhost:8501`

You should see the chatbot interface!

---

## How to Use

### Step 1: Upload a PDF Document

1. In the sidebar, click **"Upload a PDF file and start asking questions"**
2. Select a PDF file from your computer
3. Wait for the file to upload

### Step 2: Wait for Processing

After uploading:
- The app will extract text from the PDF
- Split it into chunks
- Create embeddings (this may take 1-5 minutes depending on PDF size)
- Build the vector store

**Progress indicators:**
- You'll see: "Creating embeddings and vector store..."
- Progress updates: "Embedded 10/220 chunks..."
- Success message: "✅ Vector store created successfully!"

### Step 3: Ask Questions

1. Type your question in the text input box: **"Type Your question here"**
2. Press Enter or wait a moment
3. The app will:
   - Search for relevant content in the PDF
   - Generate an answer using the LLM
   - Display the response

### Step 4: Ask Multiple Questions

- You can ask as many questions as you want
- **No need to re-upload the PDF** - embeddings are cached
- Each question is processed instantly (no re-embedding)

### Example Questions

- "What is the main topic of this document?"
- "Summarize the key points"
- "What does the document say about [topic]?"
- "List the important dates mentioned"
- "Explain the methodology used"

---

## Features

### ✅ Local Processing
- **No API keys required** - Everything runs locally
- **Privacy-first** - Your documents never leave your computer
- **Free to use** - No subscription fees

### ✅ Smart Caching
- **Embeddings cached** - PDF is processed only once
- **Fast responses** - Subsequent questions are instant
- **Session persistence** - Works across page refreshes

### ✅ Robust Error Handling
- **Automatic retries** - Handles temporary server errors
- **Progress tracking** - See embedding progress in real-time
- **Clear error messages** - Helpful troubleshooting information

### ✅ Flexible Models
- **Customizable models** - Easy to switch between Ollama models
- **Multiple embedding models** - Support for different embedding models
- **Configurable parameters** - Adjust temperature, chunk size, etc.

---

## Troubleshooting

### Problem: "Cannot connect to Ollama server"

**Solution:**
1. Check **Terminal 1** - Is `ollama serve` running?
2. If not, start it: `ollama serve`
3. Verify connection: `curl http://localhost:11434/api/tags`
4. Refresh the Streamlit app

### Problem: "ModuleNotFoundError: No module named 'X'"

**Solution:**
1. Ensure virtual environment is activated: `source gen_ai_env/bin/activate`
2. Install missing package: `pip install X`
3. Or install all dependencies: `pip install -r requirements.txt`

### Problem: "Error embedding text chunk" or "500 Server Error"

**Solution:**
1. The app will automatically retry failed chunks
2. If it persists, check Ollama server logs in **Terminal 1**
3. Restart Ollama: Stop (`Ctrl+C`) and restart (`ollama serve`)
4. Check available memory - Ollama needs sufficient RAM

### Problem: Embeddings are created every time I ask a question

**Solution:**
- This should be fixed with session state caching
- If it persists, refresh the browser page
- Make sure you're not uploading a new file each time

### Problem: Application is slow

**Possible causes:**
- Large PDF files take longer to process
- First-time embedding creation takes time
- Subsequent questions should be fast (cached)

**Solutions:**
- Wait for initial processing to complete
- Use smaller PDFs for testing
- Ensure you have sufficient RAM (8GB+ recommended)

### Problem: Models not found

**Solution:**
1. List installed models: `ollama list`
2. If models are missing, download them:
   ```bash
   ollama pull nomic-embed-text
   ollama pull llama2
   ```

### Problem: Port already in use

**Solution:**
1. Find process using port 8501: `lsof -i :8501`
2. Kill the process: `kill -9 <PID>`
3. Or use a different port: `streamlit run Chatbot-Code.py --server.port 8502`

---

## Technical Details

### Architecture

```
┌─────────────────┐
│   PDF Upload    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Text Extraction│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Text Chunking  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│   Embeddings    │◄─────┤   Ollama     │
│   (FAISS)       │      │   Server     │
└────────┬────────┘      └──────────────┘
         │
         ▼
┌─────────────────┐
│  User Question  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│ Similarity      │      │   Ollama     │
│ Search          │      │   LLM        │
└────────┬────────┘      └──────────────┘
         │
         ▼
┌─────────────────┐
│   Answer        │
└─────────────────┘
```

### Dependencies

- **streamlit**: Web application framework
- **PyPDF2**: PDF text extraction
- **langchain-ollama**: Ollama integration for LangChain
- **langchain**: LLM framework
- **langchain-community**: Community integrations
- **langchain-classic**: Legacy LangChain components
- **langchain-text-splitters**: Text splitting utilities
- **faiss-cpu**: Vector similarity search
- **requests**: HTTP requests for Ollama API

### Models Used

- **nomic-embed-text**: Embedding model (274MB)
  - Purpose: Convert text chunks to vectors
  - Context length: 8192 tokens
  - Dimensions: 768

- **llama2**: Chat model (3.8GB)
  - Purpose: Generate answers to questions
  - Parameters: 7B
  - Quantization: Q4_0

### Configuration

Key settings in `Chatbot-Code.py`:

- **Chunk size**: 1000 characters
- **Chunk overlap**: 150 characters
- **Embedding model**: `nomic-embed-text`
- **LLM model**: `llama2`
- **Temperature**: 0 (deterministic responses)
- **Ollama host**: `http://localhost:11434`

### File Structure

```
gen-ai-pdf-insights/
├── Chatbot-Code.py          # Main application file
├── README.md                # This file
├── requirements.txt         # Python dependencies (optional)
├── gen_ai_env/             # Virtual environment
│   ├── bin/
│   ├── lib/
│   └── ...
└── test_embeddings.py       # Test script (if exists)
```

---

## Quick Reference

### Terminal 1 Commands

```bash
# Start Ollama
ollama serve

# Download models
ollama pull nomic-embed-text
ollama pull llama2

# List models
ollama list

# Test connection
curl http://localhost:11434/api/tags
```

### Terminal 2 Commands

```bash
# Navigate to project
cd /Users/rithvik/projects/gen-ai-pdf-insights

# Activate virtual environment
source gen_ai_env/bin/activate

# Install dependencies
pip install streamlit PyPDF2 langchain-ollama langchain langchain-community langchain-classic langchain-text-splitters faiss-cpu requests

# Run application
streamlit run Chatbot-Code.py
```

---

## Support

If you encounter issues:

1. **Check both terminals** - Ensure both are running correctly
2. **Review error messages** - They often contain helpful information
3. **Check Ollama logs** - Look at Terminal 1 for server errors
4. **Verify models** - Run `ollama list` to confirm models are installed
5. **Restart services** - Sometimes restarting Ollama and Streamlit helps

---

## License

This project is open source and available for personal and educational use.

---

## Acknowledgments

- **Ollama** - For providing local LLM infrastructure
- **LangChain** - For the LLM framework
- **Streamlit** - For the web application framework
- **FAISS** - For efficient vector similarity search

---

**Last Updated:** November 2024

**Version:** 1.0.0
