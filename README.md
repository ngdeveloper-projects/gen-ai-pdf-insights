# Gen_AI Environment Setup and Application Instructions

## 1. Create a Project Directory

Create a folder named `Gen_AI` under the specified directory:
```bash
mkdir -p /Users/rithvik/Documents/FDP/Gen_AI
```

## 2. Environment Setup
Create and activate a Python virtual environment:

```bash
python3 -m venv gen_ai_env
source /Users/rithvik/Documents/FDP/Gen_AI/gen_ai_env/bin/activate
```

## 3. Installing Dependencies
Using pip
Install the required dependencies:

```bash
python3 -m pip install streamlit
# or to avoid conflicts with system packages
python3 -m pip install streamlit --break-system-packages

python3 -m pip install PyPDF2
# or to avoid conflicts with system packages
python3 -m pip install PyPDF2 --break-system-packages

python3 -m pip install langchain
# or to avoid conflicts with system packages
python3 -m pip install langchain --break-system-packages

python3 -m pip install langchain-community
# or to avoid conflicts with system packages
python3 -m pip install langchain-community --break-system-packages

python3 -m pip install langchain-openai
# or to avoid conflicts with system packages
python3 -m pip install langchain-openai --break-system-packages

python3 -m pip install openai
# or to avoid conflicts with system packages
python3 -m pip install openai --break-system-packages

python3 -m pip install tiktoken
# or to avoid conflicts with system packages
python3 -m pip install tiktoken --break-system-packages

python3 -m pip install faiss-cpu
# or to avoid conflicts with system packages
python3 -m pip install faiss-cpu --break-system-packages
```

## Using pipx
Alternatively, you can install the dependencies using pipx:

```bash 
brew install pipx 
pipx install streamlit
pipx install PyPDF2
pipx install langchain-openai
pipx install langchain-community
```

## Using Homebrew
For some packages, you can also use Homebrew:

```bash
brew install streamlink
brew install --cask streamlabs
```

## 4. Run the Application

To run the application, use the following command:

```bash
streamlit run Chatbot-Code.py
```





