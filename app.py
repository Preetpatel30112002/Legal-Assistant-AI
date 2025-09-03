# import streamlit as st
# import os
# import time
# import glob
# from ddgs import DDGS

# from config import (GROQ_MODEL_NAME, GROQ_API_KEY)

# from langchain_groq import ChatGroq
# from langchain_community.tools import DuckDuckGoSearchRun
# # from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain_pinecone import PineconeVectorStore
# from pinecone import Pinecone, ServerlessSpec
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_core.documents import Document

# from embed_chunks_pinecone import load_and_embed_documents_from_folder, embed_uploaded_pdf, load_and_embed_single_pdf
# from rag_pipeline import build_rag_chain

# if "messages" not in st.session_state:
#     st.session_state.messages = []
# if "rag_chain" not in st.session_state:
#     st.session_state.rag_chain = None
# if "vectorstore" not in st.session_state:
#     st.session_state.vectorstore = None
# if "embeddings" not in st.session_state:
#     st.session_state.embeddings = None
# if "file_processed" not in st.session_state:
#     st.session_state.file_processed = False
# if "selected_file" not in st.session_state:
#     st.session_state.selected_file = None
# if "selected_folder" not in st.session_state:
#     st.session_state.selected_folder = None

# @st.cache_resource
# def initialize_app_components():
#     try:
#         from rag_pipeline import initialize_components as testing_initialize
#         vectorstore, embeddings = testing_initialize()
#         return vectorstore, embeddings
#     except Exception as e:
#         st.error(f"Error while Initializing components: {str(e)}")
#         return None, None

# def stream_response(response_text):
#     result = ""
#     for char in response_text:
#         result += char
#         yield result
#         time.sleep(0.01)

# def get_pdf_files(directory):
#     pdf_files = []
#     for root, _, files in os.walk(directory):
#         for file in files:
#             if file.lower().endswith(".pdf"):
#                 pdf_files.append(os.path.join(root, file))
#     return pdf_files

# def perform_web_search(query):
#     try:
#         try:
#             with DDGS() as ddgs:
#                 results = []
#                 for r in ddgs.text(query, max_results=5):
#                     results.append(f"{r['title']}: {r['body']}")
#                 return "\n".join(results)
#         except ImportError:
#             from langchain_community.tools import DuckDuckGoSearchRun
#             web_search = DuckDuckGoSearchRun()
#             return web_search.run(query)
#     except Exception as e:
#         st.error(f"Web search error: {str(e)}")
#         return None
    
# def format_web_search_response(query, web_results):
#     try:
#         llm = ChatGroq(
#             model=GROQ_MODEL_NAME,
#             temperature=0.2,
#             max_tokens=1024,
#             api_key=GROQ_API_KEY,
#         )

#         prompt = f"""
#         You are a legal research assistant. Based on the web search results below, provide a comprehensive answer to the legal question.
        
#         QUESTION: {query}
        
#         WEB SEARCH RESULTS:
#         {web_results[:3000]}  # Limit web results to avoid token limits
        
#         INSTRUCTIONS:
#         1. Provide a complete, comprehensive answer to the question
#         2. Structure your answer with clear paragraphs
#         3. Maintain a professional, legal tone
#         4. If the search results are insufficient, acknowledge this
#         5. Do not make up information not present in the search results
        
#         ANSWER:
#         """

#         enhanced_response = llm.invoke(prompt)
#         if hasattr(enhanced_response, "content"):
#             final_answer = enhanced_response.content
#             return final_answer
#         else:
#             final_answer = str(enhanced_response)
#             return final_answer
    
#     except Exception as web_e:
#         return f"Web search results:\n\n{web_results}\n\n[Note: Could not format response due to error: {str(web_e)}]"

# def main():
#     st.set_page_config(
#         page_title = "Legal Assistant AI",
#         page_icon = "⚖️",
#         layout = "wide"
#     )

#     if st.session_state.vectorstore is None or st.session_state.embeddings is None:
#         with st.spinner("Initializing components..."):
#             vectorstore, embeddings = initialize_app_components()
#             if vectorstore and embeddings:
#                 st.session_state.vectorstore = vectorstore
#                 st.session_state.embeddings = embeddings


#     with st.sidebar:
#         st.title("⚖️ Legal Assistant")
#         st.markdown("----")

#         # st.subheader("📁 Upload PDF Documents")
#         # uploaded_file = st.file_uploader(
#         #     "Upload a legal document (PDF)",
#         #     type = "pdf",
#         #     help= "Upload PDF files to add to the Pinecone Vector Database"
#         # )

#         # if uploaded_file and st.session_state.vectorstore:
#         #     if st.button("Process PDF"):
#         #         with st.spinner("Processing PDF..."):
#         #             success, message = embed_uploaded_pdf(
#         #                 uploaded_file=uploaded_file,
#         #                 embeddings=st.session_state.embeddings,
#         #                 vectorstore=st.session_state.vectorstore
#         #             )
#         #             if success:
#         #                 st.success(message)
#         #                 st.session_state.rag_chain = build_rag_chain(st.session_state.vectorstore)
#         #                 st.session_state.file_processed = True
#         #             else:
#         #                 st.error(message)
#         # st.markdown("---")

#         st.subheader("📚 Initialize Knowledge Base")

#         input_type = st.radio(
#             "Choose input type:",
#             ["Single PDF File", "Folder with PDFs"],
#             help="Select whether to process a single PDF or all the PDFs in a folder",
#             horizontal=True
#         )

#         st.session_state.input_type = input_type

#         if input_type == "Single PDF File":
#             selected_file = st.file_uploader(
#                 "Choose a PDF file",
#                 type="pdf",
#                 key="single_file_selector"
#             )
            
#             if selected_file:
#                 st.session_state.selected_file = selected_file
#                 st.write(f"Selected: {selected_file.name}")
#         else:
#             st.info("Enter the path to a folder containing PDF files")
#             folder_path = st.text_input(
#                 "Folder path: ",
#                 placeholder="C:/Docs/legal/",
#                 key="folder_path_input"
#             )

#             if folder_path and os.path.exists(folder_path):
#                 pdf_files = get_pdf_files(folder_path)
#                 if pdf_files:
#                     st.success(f"Found {len(pdf_files)} PDF files in the folder")
#                     st.session_state.selected_folder = folder_path

#                     with st.expander("View PDF Files found"):
#                         for i, pdf_files in enumerate(pdf_files[:5]):
#                             st.write(f"{i+1}. {os.path.basename(pdf_files)}")
#                         if len(pdf_files) > 5:
#                             st.write(f"... and {len(pdf_files) - 5} more")
#                 else:
#                     st.warning("No PDF file found at the folder location")
#             elif folder_path and not os.path.exists(folder_path):
#                 st.error("❌ Folder path does not exist. Please check the path.")
        
#         can_process = (
#             (input_type == "Single PDF File" and st.session_state.selected_file is not None) or
#             (input_type == "Folder with PDFs" and st.session_state.selected_folder is not None)
#         )

#         if can_process:
#             process_btn = st.button("🔄 Initialize Knowledge Base")
#         else:
#             st.button("Initialize knowledge base", disabled=True, help="Please select a file or folder first")
#             process_btn = False

#         if process_btn:
#             if input_type == "Single PDF File":
#                 if st.session_state.selected_file is None:
#                     st.error("Please select a PDF file first.")
#                 else:
#                     with st.spinner("Processing PDF file..."):
#                         try:
#                             temp_dir = "temp_uploads"
#                             os.makedirs(temp_dir, exist_ok=True)
#                             temp_file_path = os.path.join(temp_dir, st.session_state.selected_file.name)

#                             with open(temp_file_path, "wb") as f:
#                                 f.write(st.session_state.selected_file.getbuffer())

#                             vectorstore, embeddings = load_and_embed_single_pdf(temp_file_path)
#                             try:
#                                 os.remove(temp_file_path)
#                             except:
#                                 pass
                    
#                             if vectorstore and embeddings:
#                                 st.session_state.vectorstore = vectorstore
#                                 st.session_state.embeddings = embeddings
#                                 st.session_state.rag_chain = build_rag_chain(vectorstore)
#                                 st.session_state.file_processed = True
#                                 st.success(f"Successfully processed: {st.session_state.selected_file.name}")
#                             else:
#                                 st.error("Failed to process the PDF file.")

#                         except Exception as e:
#                             st.error(f"Error Processing PDF: {str(e)}")
#             else:
#                 if st.session_state.selected_folder is None:
#                     st.error("Please specify a valid folder path first.")
#                 else:
#                     with st.spinner("Processing PDF folder..."):
#                         try:
#                             pdf_files = get_pdf_files(st.session_state.selected_folder)

#                             if not pdf_files:
#                                 st.error("No pDF files found in the specified folder.")
#                             else:

#                                 progress_bar = st.progress(0)
#                                 status_text = st.empty()

#                                 status_text.text(f"Processing {len(pdf_files)} PDF Files...")

#                                 vectorstore, embeddings = load_and_embed_documents_from_folder(st.session_state.selected_folder)

#                                 progress_bar.progress(100)
#                                 status_text.text("Processing COmpleted Successfully!")

#                                 if vectorstore and embeddings:
#                                     st.session_state.vectorstore = vectorstore
#                                     st.session_state.embeddings = embeddings
#                                     st.session_state.rag_chain = build_rag_chain(vectorstore)
#                                     st.session_state.file_processed = True
#                                     st.success(f"Successfully processed {len(pdf_files)} PDF files!")
#                                 else:
#                                     st.error("Failed to process PDF files.")
#                         except Exception as e:
#                             st.error(f"Error initializing Knowledge Base: {str(e)}")

#         if st.session_state.file_processed:
#             st.success("Knowledge base initialized successfully!")
#             if st.button("Process New Files"):
#                 st.session_state.file_processed = False
#                 st.session_state.selected_file = None
#                 st.session_state.selected_folder = None
#                 st.rerun()
                
#         st.markdown("---")
#         st.info("💡 **Tips:**\n- Ask legal questions about uploaded documents\n- The assistant uses RAG for accurate responses\n- Upload multiple PDFs to build a comprehensive knowledge base")

#     st.title("⚖️ Legal Research Assistant")
#     st.caption("Ask legal questions and get AI-powered responses with citations")

#     chat_container = st.container()

#     with chat_container:
#         for message in st.session_state.messages:
#             with st.chat_message(message["role"]):
#                 st.markdown(message["content"])

#     web_search_enabled = st.toggle(
#         "🌐 **Web Search**",
#         help="Switch ON to search the web. Switch OFF to search your documents."
#     )

#     prompt = st.chat_input("Ask a legal Question...")

#     if prompt:
#         st.session_state.messages.append({"role": "user", "content": prompt})

#         with st.chat_message("user"):
#             st.markdown(prompt)

#         with st.chat_message("assistant"):
#             message_placeholder = st.empty()
#             full_response = ""

#             try:
#                 if web_search_enabled:
#                     message_placeholder.markdown("🌐 Searching the web...")
#                     web_results = perform_web_search(prompt)

#                     if web_results:
#                         formatted_response = format_web_search_response(prompt, web_results)
#                         full_response = formatted_response
#                     else:
#                         full_response = "Sorry, I couldn't retrieve any web search results. Please try again later."
                    
#                     message_placeholder.markdown(full_response)
#                 else:   
#                     if st.session_state.rag_chain is None:
#                         error_msg = "Please initialize the knowledge base first or upload a PDF document."
#                         message_placeholder.markdown(error_msg)
#                         st.session_state.messages.append({"role":"assistant", "content": error_msg})
#                         return
                    
#                     message_placeholder.markdown("🔍 Searching legal documents...")
#                     response = st.session_state.rag_chain.invoke(prompt)
                    
#                     if isinstance(response, dict) and "output" in response:
#                         full_response = response["output"]
#                     else:
#                         full_response = str(response)
                    
#                     for chunk in stream_response(full_response):
#                         message_placeholder.markdown(chunk + "▌")
#                     message_placeholder.markdown(full_response)
                    
#                 st.session_state.messages.append({"role": "assistant", "content": full_response})
                
#             except Exception as e:
#                 error_msg = f"Sorry, I encountered an error: {str(e)}"
#                 message_placeholder.markdown(error_msg)
#                 st.session_state.messages.append({"role": "assistant", "content": error_msg})

#     col1, col2 = st.columns([1, 4])
#     with col1:
#         if st.button("🗑️ Clear Chat", use_container_width=True):
#             st.session_state.messages = []
#             st.rerun()

# if __name__ == "__main__":
#     main()


import streamlit as st
import os
import time
import glob
from ddgs import DDGS

from config import (GROQ_MODEL_NAME, GROQ_API_KEY)

from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from embed_chunks_pinecone import load_and_embed_documents_from_folder, embed_uploaded_pdf, load_and_embed_single_pdf
from rag_pipeline import build_rag_chain

if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "file_processed" not in st.session_state:
    st.session_state.file_processed = False
if "selected_file" not in st.session_state:
    st.session_state.selected_file = None
if "selected_folder" not in st.session_state:
    st.session_state.selected_folder = None

@st.cache_resource
def initialize_app_components():
    try:
        from rag_pipeline import initialize_components as testing_initialize
        vectorstore, embeddings = testing_initialize()
        return vectorstore, embeddings
    except Exception as e:
        st.error(f"Error while Initializing components: {str(e)}")
        return None, None

def stream_response(response_text):
    result = ""
    for char in response_text:
        result += char
        yield result
        time.sleep(0.01)

def get_pdf_files(directory):
    pdf_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def perform_web_search(query):
    try:
        try:
            with DDGS() as ddgs:
                results = []
                for r in ddgs.text(query, max_results=5):
                    results.append(f"{r['title']}: {r['body']}")
                return "\n".join(results)
        except ImportError:
            from langchain_community.tools import DuckDuckGoSearchRun
            web_search = DuckDuckGoSearchRun()
            return web_search.run(query)
    except Exception as e:
        st.error(f"Web search error: {str(e)}")
        return None
    
def format_web_search_response(query, web_results):
    try:
        llm = ChatGroq(
            model=GROQ_MODEL_NAME,
            temperature=0.2,
            max_tokens=1024,
            api_key=GROQ_API_KEY,
        )

        prompt = f"""
        You are a legal research assistant. Based on the web search results below, provide a comprehensive answer to the legal question.
        
        QUESTION: {query}
        
        WEB SEARCH RESULTS:
        {web_results[:3000]}  # Limit web results to avoid token limits
        
        INSTRUCTIONS:
        1. Provide a complete, comprehensive answer to the question
        2. Structure your answer with clear paragraphs
        3. Maintain a professional, legal tone
        4. If the search results are insufficient, acknowledge this
        5. Do not make up information not present in the search results
        
        ANSWER:
        """

        enhanced_response = llm.invoke(prompt)
        if hasattr(enhanced_response, "content"):
            final_answer = enhanced_response.content
            return final_answer
        else:
            final_answer = str(enhanced_response)
            return final_answer
    
    except Exception as web_e:
        return f"Web search results:\n\n{web_results}\n\n[Note: Could not format response due to error: {str(web_e)}]"

def main():
    st.set_page_config(
        page_title = "Legal Assistant AI",
        page_icon = "⚖️",
        layout = "wide"
    )

    if st.session_state.vectorstore is None or st.session_state.embeddings is None:
        with st.spinner("Initializing components..."):
            vectorstore, embeddings = initialize_app_components()
            if vectorstore and embeddings:
                st.session_state.vectorstore = vectorstore
                st.session_state.embeddings = embeddings


    # PASTE THIS NEW CODE BLOCK TO REPLACE THE ENTIRE 'with st.sidebar:' SECTION IN app.py

    with st.sidebar:
        st.title("⚖️ Legal Assistant")
        st.markdown("----")

        st.subheader("📚 Select Document Source")

        input_type = st.radio(
            "Choose input type:",
            ["Single PDF File", "Folder with PDFs"],
            help="Select whether to process a single PDF or all PDFs in a folder.",
            horizontal=True
        )

        st.session_state.input_type = input_type

        if input_type == "Single PDF File":
            selected_file = st.file_uploader(
                "Choose a PDF file to process",
                type="pdf",
                key="single_file_selector"
            )
            
            if selected_file:
                st.session_state.selected_file = selected_file
                st.write(f"Selected: `{selected_file.name}`")
                
                # Add process button for single file
                if st.button("🔄 Process PDF", use_container_width=True):
                    with st.spinner("Processing PDF file..."):
                        try:
                            temp_dir = "temp_uploads"
                            os.makedirs(temp_dir, exist_ok=True)
                            temp_file_path = os.path.join(temp_dir, selected_file.name)

                            with open(temp_file_path, "wb") as f:
                                f.write(selected_file.getbuffer())

                            vectorstore, embeddings = load_and_embed_single_pdf(temp_file_path)
                            
                            try:
                                os.remove(temp_file_path)
                            except:
                                pass
                    
                            if vectorstore and embeddings:
                                st.session_state.vectorstore = vectorstore
                                st.session_state.embeddings = embeddings
                                st.session_state.rag_chain = build_rag_chain(vectorstore)
                                st.session_state.file_processed = True
                                st.success(f"Successfully processed: {selected_file.name}")
                            else:
                                st.error("Failed to process the PDF file.")

                        except Exception as e:
                            st.error(f"Error Processing PDF: {str(e)}")
            else:
                st.session_state.selected_file = None
                st.session_state.file_processed = False
        
        else:  # Folder with PDFs
            st.info("Enter the path to a folder containing PDF files.")
            folder_path = st.text_input(
                "Folder path:",
                placeholder="C:/Docs/legal/",
                key="folder_path_input"
            )

            if folder_path and os.path.exists(folder_path):
                pdf_files = get_pdf_files(folder_path)
                if pdf_files:
                    st.success(f"Found {len(pdf_files)} PDF files.")
                    st.session_state.selected_folder = folder_path
                    
                    with st.expander("View PDF Files found"):
                        for i, pdf_file in enumerate(pdf_files[:5]):
                            st.write(f"{i+1}. {os.path.basename(pdf_file)}")
                        if len(pdf_files) > 5:
                            st.write(f"... and {len(pdf_files) - 5} more")
                    
                    # Add process button for folder
                    if st.button("🔄 Process Folder", use_container_width=True):
                        with st.spinner(f"Processing {len(pdf_files)} PDF files..."):
                            try:
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                status_text.text(f"Processing {len(pdf_files)} PDF Files...")

                                vectorstore, embeddings = load_and_embed_documents_from_folder(folder_path)
                                
                                progress_bar.progress(100)
                                status_text.text("Processing Completed Successfully!")
                                
                                if vectorstore and embeddings:
                                    st.session_state.vectorstore = vectorstore
                                    st.session_state.embeddings = embeddings
                                    st.session_state.rag_chain = build_rag_chain(vectorstore)
                                    st.session_state.file_processed = True
                                    st.success(f"Successfully processed {len(pdf_files)} PDF files!")
                                else:
                                    st.error("Failed to process PDF files.")
                            except Exception as e:
                                st.error(f"Error processing folder: {str(e)}")
                else:
                    st.warning("No PDF files found in the specified folder.")
                    st.session_state.selected_folder = None
                    st.session_state.file_processed = False
            elif folder_path:
                st.error("Folder path does not exist.")
                st.session_state.selected_folder = None
                st.session_state.file_processed = False
            else:
                st.session_state.selected_folder = None
                st.session_state.file_processed = False

        st.markdown("---")
        
        # Show processing status
        if st.session_state.file_processed:
            st.success("✅ Knowledge base ready!")
            if st.button("🔄 Process New Files", use_container_width=True):
                st.session_state.file_processed = False
                st.session_state.selected_file = None
                st.session_state.selected_folder = None
                st.rerun()
        
        st.info("💡 **Tips:**\n- Select a file or folder above\n- Click the process button to prepare documents\n- Use web search for general legal questions\n- Use document search for uploaded content")

    st.title("⚖️ Legal Research Assistant")
    st.caption("Ask legal questions and get AI-powered responses with citations")

    chat_container = st.container()

    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    web_search_enabled = st.toggle(
        "🌐 **Web Search**",
        help="Switch ON to search the web. Switch OFF to search your documents."
    )

    prompt = st.chat_input("Ask a legal Question...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                if web_search_enabled:
                    message_placeholder.markdown("🌐 Searching the web...")
                    web_results = perform_web_search(prompt)

                    if web_results:
                        formatted_response = format_web_search_response(prompt, web_results)
                        full_response = formatted_response
                    else:
                        full_response = "Sorry, I couldn't retrieve any web search results. Please try again later."
                    
                    message_placeholder.markdown(full_response)
                else:   
                    if st.session_state.rag_chain is None:
                        error_msg = "Please initialize the knowledge base first or upload a PDF document."
                        message_placeholder.markdown(error_msg)
                        st.session_state.messages.append({"role":"assistant", "content": error_msg})
                        return
                    
                    message_placeholder.markdown("🔍 Searching legal documents...")
                    response = st.session_state.rag_chain.invoke(prompt)
                    
                    if isinstance(response, dict) and "output" in response:
                        full_response = response["output"]
                    else:
                        full_response = str(response)
                    
                    for chunk in stream_response(full_response):
                        message_placeholder.markdown(chunk + "▌")
                    message_placeholder.markdown(full_response)
                    
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                message_placeholder.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()


