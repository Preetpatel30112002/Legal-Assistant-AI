from typing import Any, List, Tuple
import torch
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun

from config import (
    PINECONE_API_KEY,
    EMBEDDING_MODEL_NAME,
    PINECONE_INDEX_NAME,
    TOP_K,
    GROQ_MODEL_NAME,
    GROQ_API_KEY
)

def render_sources(docs_with_scores: List[Tuple[Any, float]]):
    if not docs_with_scores:
        return "No sources found."
    lines = []
    for i, (doc, score) in enumerate(docs_with_scores, start=1):
        meta = doc.metadata or {}
        case_name = meta.get("case_name", "Unknown")
        src = meta.get("source", "unknown")
        page_snippets = (doc.page_content or "").strip().replace("\n", " ")

        if len(page_snippets) > 200:
            page_snippets = page_snippets[:240] + " ..."
        lines.append(
            f"[{i}] score={score:.4f} | case={case_name} | src={src}\n      {page_snippets}"
        )

    return "\n".join(lines)

def initialize_components():
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)

        # Better device handling
        device = "cpu"  # Default to CPU
        if torch.cuda.is_available():
            try:
                # Test if CUDA actually works
                test_tensor = torch.tensor([1.0]).cuda()
                device = "cuda"
            except:
                device = "cpu"
                print("CUDA available but not working, falling back to CPU")

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": device},
            encode_kwargs = {"normalize_embeddings": True},
        )

        vectorstore = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings,
        )

        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            print(f"Warning: Index {PINECONE_INDEX_NAME} does not exist yet")
            return None, None
        
        return vectorstore, embeddings
    except Exception as e:
        print(f"Error initializing components: {str(e)}")
        return None, None

def build_rag_chain(vectorstore, model: str = GROQ_MODEL_NAME):
    try:
        llm_model = ChatGroq(
            model_name=model,
            temperature=0.2,
            max_tokens=1024,
            groq_api_key=GROQ_API_KEY,
        )

        system_msg = (
            "You are a legal research assistant. You MUST follow these rules:\n"
            "1. Answer using ONLY the provided context from legal documents\n"
            "2. If the answer cannot be found in the context, say 'I cannot find this information in the provided legal documents.'\n"
            "3. NEVER use your general knowledge about cases - only use the context provided\n"
            "4. Be concise and precise\n"
            "5. At the end, include citations to the specific documents used with case names and sources"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_msg),
                (
                    "user",
                    "Question:\n{question}\n\nContext from legal documents:\n{context}\n\n"
                    "Return a direct answer based ONLY on the context above. If the answer isn't there, say so.\n"
                ),
            ]
        )

        def format_docs(docs):
            return "\n\n".join(f"[{i}] {doc.page_content}" for i, doc in enumerate(docs, 1))
        
        retriever = vectorstore.as_retriever(
            search_type = "similarity",
            search_kwargs = {"k": TOP_K}
        )

        chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm_model
            | StrOutputParser()
        )

        return chain
    except Exception as e:
        print(f"Error Building RAG Chain: {str(e)}")
        return None
    
def build_legal_assistant_agent(vectorstore):
    llm = ChatGroq(
        model_name=GROQ_MODEL_NAME,
        temperature=0.2,
        max_tokens=1024,
        groq_api_key=GROQ_API_KEY,
    )

    rag_tool = build_rag_chain(vectorstore, GROQ_MODEL_NAME)

    @tool
    def legal_document_search_tool(question: str) -> str:
        """Searches uploaded legal documents for answers to a given question.
        Returns 'NO_MATCH' if no relevant answer is found."""
        try:
            response = rag_tool.invoke(question)
            if not response or response.strip() == "" or "cannot find this information" in response.lower():
                return "NO_MATCH"
            return response
        except Exception as e:
            return f"Error searching legal documents: {str(e)}"

    @tool
    def fallback_for_websearch(query: str)->str:
        """Fallback web search tool using DuckDuckGo.
        Used ONLY if the legal document search tool returns 'NO_MATCH'."""
        try:
            search = DuckDuckGoSearchRun()
            return search.run(query)
        except Exception as e:
            return f"Error while running web search: {str(e)}"

    # web_search_tool = DuckDuckGoSearchRun(
    #     name="web_search",
    #     description="Fallback search for general legal info or when RAG cannot find the answer."
    # )

    tools = [legal_document_search_tool, fallback_for_websearch]

    # In build_legal_assistant_agent function, improve the system prompt:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a legal research assistant specializing in Indian case law.

        CRITICAL INSTRUCTIONS:
        1. ALWAYS use legal_document_search_tool FIRST for any question about cases, people, or legal matters
        2. ONLY use web_search if legal_document_search_tool returns 'NO_MATCH' EXACTLY
        3. If legal_document_search_tool returns ANY information (even if it says "role not specified"), use that information
        4. When documents mention a person but don't specify their exact role, summarize what information IS available
        5. Never say "could not be found" if the documents mention the person at all
        6. Always cite the specific case name and document source

        EXAMPLE:
        User: "Who is Anil Kumar Yadav in Anil_Kumar_Yadav_vs_State_Of_Nct_Delhi case?"
        Assistant: [Uses legal_document_search_tool, gets response about custody duration and bail]
        Assistant: "Based on the case documents, while Anil Kumar Yadav's specific role isn't explicitly stated, the documents reveal he was accused A4 who had been in custody for about sixteen months. The case involved [other details found]." 
        """),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    return agent_executor
