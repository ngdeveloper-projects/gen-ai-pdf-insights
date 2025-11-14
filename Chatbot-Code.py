import os
# CRITICAL: Set Ollama host BEFORE importing any Ollama-related modules
# This must be done before any imports that use Ollama
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"
os.environ["OLLAMA_BASE_URL"] = "http://127.0.0.1:11434"

import streamlit as st
import requests
import time
import re
from typing import List, Tuple, Optional
from PyPDF2 import PdfReader
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.embeddings import Embeddings

def extract_page_number_from_footer(page_text: str, page_index: int) -> Optional[int]:
    """
    Extract page number from PDF footer text.
    Tries multiple strategies to find the actual page number from footer.
    """
    if not page_text:
        return None
    
    # Strategy 1: Look for common footer patterns (e.g., "Page 5", "5", "Page 5 of 10")
    # Check bottom portion of text (last 30% of lines, as footer is usually at bottom)
    lines = page_text.split('\n')
    if len(lines) > 3:
        # Take last 30% of lines, but at least last 3 lines
        footer_lines = lines[-max(3, len(lines)//3):]
    else:
        footer_lines = lines
    
    footer_text = '\n'.join(footer_lines)
    
    # Also check the very last line separately (most common footer position)
    last_line = lines[-1] if lines else ""
    
    # Patterns to try (in order of specificity)
    patterns = [
        # Most specific patterns first
        (r'[Pp]age\s+(\d+)\s+of\s+\d+', last_line),  # "Page 5 of 10" on last line
        (r'[Pp]age\s+(\d+)', footer_text),  # "Page 5" anywhere in footer
        (r'^(\d+)\s*/\s*\d+$', last_line),  # "5/10" format on last line
        (r'^(\d+)\s+of\s+\d+$', last_line),  # "5 of 10" format on last line
        (r'^(\d+)$', last_line),  # Just a number on its own line (last line)
        (r'\b(\d{1,4})\b', footer_text),  # Any 1-4 digit number in footer (avoid years/dates)
    ]
    
    for pattern, text_to_search in patterns:
        matches = re.findall(pattern, text_to_search, re.MULTILINE)
        if matches:
            # Take the last match (footer usually has page number at the end)
            try:
                page_num = int(matches[-1])
                # Sanity check: page number should be reasonable
                # Exclude common non-page numbers (years, dates, etc.)
                if 1 <= page_num <= 10000:  # Reasonable range
                    # Additional check: if it's a 4-digit number starting with 19 or 20, it's likely a year
                    if len(str(page_num)) == 4 and (str(page_num).startswith('19') or str(page_num).startswith('20')):
                        continue
                    return page_num
            except ValueError:
                continue
    
    # Fallback: return None if we can't find a page number
    return None

def extract_page_numbers_from_pdf(pdf_reader: PdfReader) -> List[Tuple[str, int, Optional[int]]]:
    """
    Extract text and page numbers from PDF.
    Returns list of (page_text, page_index, footer_page_number) tuples.
    """
    text_with_pages = []
    
    for page_index, page in enumerate(pdf_reader.pages, start=0):
        page_text = page.extract_text()
        if page_text.strip():
            # Try to extract page number from footer
            footer_page_num = extract_page_number_from_footer(page_text, page_index)
            text_with_pages.append((page_text, page_index + 1, footer_page_num))
    
    return text_with_pages

# Custom embedding class that uses direct HTTP requests to Ollama API
class OllamaEmbeddingsWrapper(Embeddings):
    """Custom embedding class that uses direct HTTP requests to avoid port issues"""
    
    def __init__(self, model: str = "nomic-embed-text", host: str = "http://127.0.0.1:11434"):
        self.model = model
        self.host = host.rstrip('/')
        self.api_url = f"{self.host}/api/embeddings"
        # Ensure environment variable is set
        os.environ["OLLAMA_HOST"] = host
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using direct HTTP requests with retry logic"""
        embeddings = []
        total = len(texts)
        max_retries = 3
        base_retry_delay = 2  # seconds
        request_delay = 0.1  # Small delay between requests to avoid overwhelming server
        
        for i, text in enumerate(texts):
            # Truncate text if it's too long (embedding models have token limits)
            # nomic-embed-text supports up to 8192 tokens, but let's be safe
            max_length = 4000  # characters (roughly 1000 tokens)
            if len(text) > max_length:
                text = text[:max_length]
                st.warning(f"Chunk {i+1} was truncated to {max_length} characters")
            
            # Retry logic for failed requests
            retry_delay = base_retry_delay
            success = False
            
            for attempt in range(max_retries):
                try:
                    # Use direct HTTP request to ensure we connect to the correct port
                    payload = {
                        "model": self.model,
                        "prompt": text
                    }
                    try:
                        response = requests.post(self.api_url, json=payload, timeout=300)
                        response.raise_for_status()
                        result = response.json()
                        embeddings.append(result['embedding'])
                        success = True
                        break  # Success, exit retry loop
                    except requests.exceptions.HTTPError as e:
                        if response.status_code == 500 and attempt < max_retries - 1:
                            # Server error, wait and retry
                            st.warning(f"Server error on chunk {i+1}, attempt {attempt+1}/{max_retries}. Retrying in {retry_delay}s...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                            continue
                        else:
                            raise
                except Exception as e:
                    if attempt == max_retries - 1:
                        # Last attempt failed
                        st.error(f"Error embedding text chunk {i+1}/{total} after {max_retries} attempts: {e}")
                        st.error(f"Chunk length: {len(text)} characters")
                        st.error(f"Trying to connect to: {self.api_url}")
                        # Skip this chunk and continue with others
                        st.warning(f"Skipping chunk {i+1} and continuing...")
                        # Add a zero vector as placeholder (same dimension as other embeddings)
                        if embeddings:
                            embeddings.append([0.0] * len(embeddings[0]))
                        else:
                            # If this is the first chunk, we can't continue
                            raise
                    else:
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
            
            # Small delay between requests to avoid overwhelming the server
            if success and i < total - 1:  # Don't delay after the last chunk
                time.sleep(request_delay)
            
            # Show progress for large batches
            if total > 10 and (i + 1) % 10 == 0:
                st.info(f"Embedded {i + 1}/{total} chunks...")
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using direct HTTP request"""
        try:
            # Use direct HTTP request to ensure we connect to the correct port
            payload = {
                "model": self.model,
                "prompt": text
            }
            response = requests.post(self.api_url, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()
            return result['embedding']
        except Exception as e:
            st.error(f"Error embedding query: {e}")
            st.error(f"Trying to connect to: {self.api_url}")
            raise

# Initialize session state for vector store (must be before any usage)
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "file_processed" not in st.session_state:
    st.session_state.file_processed = None
if "chunks_with_metadata" not in st.session_state:
    st.session_state.chunks_with_metadata = []
if "total_pages" not in st.session_state:
    st.session_state.total_pages = 0

#Upload PDF files
st.header("My first Chatbot")

with  st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader(" Upload a PDF file and start asking questions", type="pdf")
    
    # Show current file status
    if st.session_state.file_processed is not None:
        st.success("✅ PDF loaded and ready for questions")
    elif file is not None:
        st.info("⏳ Processing PDF...")

#Extract the text
if file is not None:
    # Check if this is a new file (different from the one we processed)
    file_id = file.name + str(file.size) + str(file.type)
    
    # Only process if it's a new file
    if st.session_state.file_processed != file_id:
        pdf_reader = PdfReader(file)
        total_pages = len(pdf_reader.pages)
        st.session_state.total_pages = total_pages
        
        # Extract text with page numbers from footer
        text_with_pages = extract_page_numbers_from_pdf(pdf_reader)
        
        # Combine all text for chunking
        full_text = "\n".join([text for text, _, _ in text_with_pages])

        #Break it into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators="\n",
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )
        chunks = text_splitter.split_text(full_text)
        
        # Map chunks to page numbers (using footer page numbers when available)
        chunks_with_metadata = []
        current_pos = 0
        
        for chunk in chunks:
            # Find which page this chunk likely belongs to
            chunk_start = current_pos
            chunk_end = current_pos + len(chunk)
            
            # Determine page number based on text position
            page_num = 1  # Default fallback
            footer_page_num = None
            page_index = 1  # Default sequential index
            cumulative_length = 0
            
            for page_text, seq_index, footer_num in text_with_pages:
                cumulative_length += len(page_text) + 1  # +1 for newline
                if chunk_start < cumulative_length:
                    # Use footer page number if available, otherwise use sequential index
                    if footer_num is not None:
                        page_num = footer_num
                        footer_page_num = footer_num
                    else:
                        page_num = seq_index
                    page_index = seq_index
                    break
            
            chunks_with_metadata.append({
                "text": chunk,
                "page": page_num,
                "page_index": page_index,
                "footer_page": footer_page_num,
                "chunk_index": len(chunks_with_metadata)
            })
            current_pos = chunk_end
        
        # Store chunks with metadata
        st.session_state.chunks_with_metadata = chunks_with_metadata
        #st.write(chunks)

        # generating embedding using Ollama
        # Using nomic-embed-text model for embeddings
        # Ensure environment variable is set (in case Streamlit reset it)
        os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"
        
        try:
            # Test connection first
            test_response = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
            if test_response.status_code != 200:
                st.error(f"Ollama server not responding correctly. Status: {test_response.status_code}")
                st.stop()
            
            # Initialize embeddings using our custom wrapper that properly sets the host
            embeddings = OllamaEmbeddingsWrapper(
                model="nomic-embed-text",
                host="http://127.0.0.1:11434"
            )
        except requests.exceptions.RequestException as e:
            st.error("❌ **Cannot connect to Ollama server**")
            st.error(f"Error: {e}")
            st.warning("""
            **To fix this, please:**
            
            1. Open a **new terminal window**
            2. Run the following command:
               ```bash
               ollama serve
               ```
            3. Keep that terminal window open (Ollama needs to keep running)
            4. Come back here and refresh this page
            
            **Note:** Ollama must be running in a separate terminal before you can use this app.
            """)
            st.stop()
        except Exception as e:
            st.error(f"Error initializing embeddings: {e}")
            st.error("Make sure Ollama is running on port 11434")
            st.error(f"Current OLLAMA_HOST: {os.environ.get('OLLAMA_HOST', 'Not set')}")
            st.stop()

        # creating vector store - FAISS (only once per file)
        try:
            with st.spinner("Creating embeddings and vector store (this only happens once per file)..."):
                vector_store = FAISS.from_texts(chunks, embeddings)
                st.session_state.vector_store = vector_store
                st.session_state.file_processed = file_id
                st.success("✅ Vector store created successfully! You can now ask questions.")
        except Exception as e:
            st.error(f"Error creating vector store: {e}")
            st.error("Make sure Ollama is running and the 'nomic-embed-text' model is available.")
            st.stop()
    else:
        # File already processed, use cached vector store
        st.info("📄 Using cached vector store. You can ask questions now!")

    # get user question (only show if PDF is uploaded and processed)
    if st.session_state.vector_store is not None:
        # Add checkboxes for options
        col1, col2 = st.columns(2)
        with col1:
            show_sources = st.checkbox("📄 Show sources", value=False, help="Display the source sections from the PDF used to generate the answer")
        with col2:
            include_model_answer = st.checkbox("🤖 Include model answer", value=False, help="If PDF doesn't contain information, generate answer from model knowledge")
        
        user_question = st.text_input("Type Your question here")

        # Ensure we only process questions when PDF is uploaded and vector store exists
        if user_question:
            # Show loading spinner while processing the question
            with st.spinner("🔍 Searching the PDF and generating answer..."):
                # do similarity search using the uploaded PDF's vector store
                # Get more results if showing sources
                k = 5 if show_sources else 3
                match = st.session_state.vector_store.similarity_search(user_question, k=k)
                
                # Check if the matched content actually contains relevant information
                # Extract keywords from the question
                question_keywords = set(word.lower() for word in user_question.split() if len(word) > 3)
                
                # Check if any matched document contains the keywords
                relevant_found = False
                matched_text = ""
                for doc in match:
                    doc_text_lower = doc.page_content.lower()
                    matched_text += doc.page_content + "\n\n"
                    # Check if at least one keyword appears in the matched content
                    for keyword in question_keywords:
                        if keyword in doc_text_lower:
                            relevant_found = True
                            break
                    if relevant_found:
                        break
                
                # If no relevant information found, handle based on checkbox
                if not relevant_found and len(question_keywords) > 0:
                    st.markdown("### 💬 Answer:")
                    st.warning("❌ **No information found in the uploaded PDF**")
                    
                    # If "include model answer" is checked, generate answer from model
                    if include_model_answer:
                        st.info("⚠️ **Note:** The following answer is generated from the model's general knowledge, NOT from your uploaded PDF.")
                        st.markdown("---")
                        
                        # Generate answer from model without PDF context
                        llm = ChatOllama(
                            model="llama2",
                            temperature=0,
                            base_url="http://127.0.0.1:11434"
                        )
                        
                        # Create a simple prompt that indicates this is general knowledge
                        general_prompt = f"""Answer the following question based on your general knowledge. 
Note: This question could not be answered from the provided document, so provide a general answer.

Question: {user_question}

Answer:"""
                        
                        with st.spinner("Generating answer from model knowledge..."):
                            response = llm.invoke(general_prompt)
                            # Extract text from response (ChatOllama returns AIMessage object)
                            if hasattr(response, 'content'):
                                model_answer = response.content
                            elif hasattr(response, 'text'):
                                model_answer = response.text
                            else:
                                model_answer = str(response)
                        
                        # Display model answer with clear warning
                        st.markdown("**📝 Model Answer (Not from PDF):**")
                        st.markdown("---")
                        st.write(model_answer)
                        st.markdown("---")
                        st.warning("⚠️ **Important:** This answer is from the AI model's general knowledge, not from your uploaded PDF document.")
                    else:
                        st.info("The PDF does not contain information related to your question. Please try asking about topics that are covered in the document.")
                        st.info("💡 **Tip:** Enable 'Include model answer' checkbox to get an answer from the model's general knowledge even when the PDF doesn't contain relevant information.")
                    
                    # Show source information
                    st.markdown("---")
                    st.caption(f"📄 Searched through {len(match)} section(s) from your uploaded PDF")
                    
                    # Show sources if checkbox is checked
                    if show_sources:
                        st.markdown("### 📚 Sources:")
                        st.info("No relevant sources found in the PDF for this query.")
                        st.markdown("**Searched sections:**")
                        for i, doc in enumerate(match[:3], 1):  # Show first 3 matches
                            with st.expander(f"Searched Section {i}", expanded=False):
                                st.text_area(
                                    "Content:",
                                    value=doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                                    height=150,
                                    key=f"searched_{i}",
                                    label_visibility="collapsed",
                                    disabled=True
                                )
                else:
                    # Continue with normal processing if relevant information found
                    #define the LLM using Ollama
                    # You can change the model to llama2, mistral, or any other Ollama model you have installed
                    llm = ChatOllama(
                        model="llama2",
                        temperature=0,
                        base_url="http://127.0.0.1:11434"
                    )

                    #output results
                    #chain -> take the question, get relevant document from PDF, pass it to the LLM, generate the output
                    chain = load_qa_chain(llm, chain_type="stuff")
                    response = chain.run(input_documents = match, question = user_question)
                
                    # Display the answer
                    st.markdown("### 💬 Answer:")
                    st.write(response)
                    
                    # Show source information
                    st.markdown("---")
                    st.caption(f"📄 Based on {len(match)} relevant section(s) from your uploaded PDF")
                    
                    # Show sources if checkbox is checked
                    if show_sources:
                        st.markdown("### 📚 Sources:")
                        
                        # Find matching chunks with metadata
                        source_info = []
                        seen_chunks = set()  # To avoid duplicates
                        
                        for doc in match:
                            doc_text = doc.page_content.strip()
                            best_match = None
                            best_similarity = 0
                            
                            # Try to find the best matching chunk
                            for chunk_meta in st.session_state.chunks_with_metadata:
                                chunk_text = chunk_meta["text"].strip()
                                
                                # Calculate similarity (simple character overlap)
                                # Check if significant portion of text matches
                                overlap = min(len(doc_text), len(chunk_text))
                                if overlap > 100:  # Only check if both are substantial
                                    # Check for substring match
                                    if doc_text[:min(200, len(doc_text))] in chunk_text or \
                                       chunk_text[:min(200, len(chunk_text))] in doc_text:
                                        similarity = min(len(doc_text), len(chunk_text)) / max(len(doc_text), len(chunk_text))
                                        if similarity > best_similarity and chunk_meta["chunk_index"] not in seen_chunks:
                                            best_similarity = similarity
                                            best_match = chunk_meta
                            
                            # Add best match if found
                            if best_match and best_match["chunk_index"] not in seen_chunks:
                                source_info.append({
                                    "full_text": best_match["text"],
                                    "chunk_index": best_match["chunk_index"]
                                })
                                seen_chunks.add(best_match["chunk_index"])
                        
                        # Display sources with high-level information (no page numbers)
                        if source_info:
                            # Count occurrences of keywords in sources
                            question_keywords = [word.lower() for word in user_question.split() if len(word) > 3]
                            occurrence_count = 0
                            for source in source_info:
                                source_lower = source["full_text"].lower()
                                for keyword in question_keywords:
                                    occurrence_count += source_lower.count(keyword)
                            
                            # Show high-level summary
                            if occurrence_count > 0:
                                st.success(f"✅ **Sources found:** {len(source_info)} relevant section(s) with {occurrence_count} occurrence(s) of your query terms")
                            else:
                                st.info(f"📄 **Sources found:** {len(source_info)} relevant section(s) from the PDF")
                            
                            # Show source content without page numbers
                            for i, source in enumerate(source_info, 1):
                                with st.expander(f"Source {i} of {len(source_info)}", expanded=False):
                                    st.text_area(
                                        "Content:",
                                        value=source['full_text'],
                                        height=200,
                                        key=f"source_{i}_{source['chunk_index']}",
                                        label_visibility="collapsed",
                                        disabled=True
                                    )
                                    st.caption(f"Section {source['chunk_index'] + 1} from the PDF")
                        else:
                            # Fallback: show matched content
                            st.info(f"📄 **Sources found:** {len(match)} relevant section(s) from the PDF")
                            for i, doc in enumerate(match, 1):
                                with st.expander(f"Source {i} of {len(match)}", expanded=False):
                                    st.text_area(
                                        "Content:",
                                        value=doc.page_content,
                                        height=200,
                                        key=f"match_{i}",
                                        label_visibility="collapsed",
                                        disabled=True
                                    )
    else:
        # Show message if no PDF is uploaded yet
        st.info("👆 Please upload a PDF file first to start asking questions.")