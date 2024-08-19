1. Create a Folder named "Gen_AI" under "/Users/rithvik/Documents/FDP/"
2. Env Setup:
	python3 -m venv gen_ai_env
	source /Users/rithvik/Documents/FDP/Gen_AI/gen_ai_env/bin/activate
3. Installing Dependencies:
	pipx install streamlit
	(or)
	brew install streamlink
	brew install --cask streamlabs
	(or)
	python3 -m pip install streamlit (or) python3 -m pip install streamlit --break-system-packages
	python3 -m pip install PyPDF2 (or) python3 -m pip install PyPDF2 --break-system-packages
	python3 -m pip install langchain (or) python3 -m pip install langchain --break-system-packages
	python3 -m pip install langchain-community (or) python3 -m pip install langchain-community --break-system-packages
	python3 -m pip install langchain-openai (or) python3 -m pip install langchain-openai --break-system-packages
	python3 -m pip install openai (or) python3 -m pip install openai --break-system-packages
	python3 -m pip install tiktoken (or) python3 -m pip install tiktoken --break-system-packages
	python3 -m pip install faiss-cpu (or) python3 -m pip install faiss-cpu --break-system-packages 
	
	Installation using pipx:
	brew install pipx 
	pipx install streamlit                                                      
	pipx install PyPDF2
	pipx install langchain-openai
	pipx install langchain-community

4. Run the application:
	streamlit run Chatbot-Code.py
