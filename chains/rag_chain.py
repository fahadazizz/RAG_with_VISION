from typing import Optional, Generator, List
from dataclasses import dataclass

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from chains.retriever import get_retriever
from models.llm import get_ollama_llm
from prompts.rag_prompts import get_rag_prompt


@dataclass
class RAGResponse:
    answer: str
    sources: List[dict]
    query: Optional[str]
    
    def __str__(self) -> str:
        return self.answer


class RAGChain:
    
    def __init__(self):
        self._retriever = get_retriever()
        self._llm = get_ollama_llm()
        self._prompt = get_rag_prompt()
        
        self._chain = self._build_chain()
    
    def _build_chain(self):
        """
        Build the LCEL chain for RAG.
        
        Returns:
            LCEL chain = prompted LLM
        """
        # We pass "context", "input_image_context", and "question" directly to invoke()
        chain = (
            self._prompt
            | self._llm._llm
            | StrOutputParser()
        )
        
        return chain
    
    def query(
        self,
        question: Optional[str],
        image_query_path: Optional[str] = None,
    ) -> RAGResponse:
        """
        Execute a RAG query and get a response.
        """
        # 0. Analyze Query & Image
        search_query = question
        is_instruction = False
        
        if question:
            try:
                from chains.query_analyzer import get_query_analyzer
                analyzer = get_query_analyzer()
                print(f"Analyzing query: '{question}'")
                analysis = analyzer.analyze(question)
                
                if image_query_path and analysis.is_instruction:
                    # If valid image + instruction -> Ignore text for retrieval
                    search_query = None 
                    print(f"Query Analysis: Treated '{question}' as INSTRUCTION. Search Query: None")
                elif analysis.search_query:
                    # Use extracted search query
                    search_query = analysis.search_query
                    print(f"Query Analysis: Refined Search Query: '{search_query}'")
            except Exception as e:
                print(f"Query analyzer failed, falling back to original query: {e}")

        # Get Image Label/Description for the Prompt
        input_image_context = "No input image provided."
        if image_query_path:
            try:
                from models.clip_model import get_clip_model
                clip = get_clip_model()
                # candidates extended for better description
                candidates = ["chart", "diagram", "table", "screenshot of code", "photograph", "document page", "plot", "graph", "infographic", "natural image", "software interface"]
                label = clip.get_image_label(image_query_path, candidates)
                input_image_context = f"The user provided an image classified as: {label}."
                print(f"Input Image Context: {input_image_context}")
            except Exception as e:
                print(f"Error getting image label: {e}")
                input_image_context = "User provided an image, but classification failed."

        # 1. Retrieve documents (Multimodal if image provided)
        # Use refined search_query
        results = self._retriever.retrieve(
            query=search_query if search_query else "",
            image_query_path=image_query_path
        )
        
        # 2. Format context
        context = self._retriever.format_context(results)
        
        # 3. Extract sources
        sources = [result.metadata for result in results]
        
        # 4. Generate answer
        # Ensure question is not None for the LLM prompt
        safe_question = question if question else "Analyze the provided input image context."
        
        response = self._chain.invoke({
            "context": context,
            "input_image_context": input_image_context,
            "question": safe_question
        })
        
        return RAGResponse(
            answer=response,
            sources=sources,
            query=question,
        )
    
    def stream_query(
        self,
        question: Optional[str],
        image_query_path: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Stream a RAG query response.
        """
        # Logic duplicated for streaming - simplified for now to match structure
        # Ideally refactor to shared method
        
        search_query = question
        if question:
             try:
                from chains.query_analyzer import get_query_analyzer
                analyzer = get_query_analyzer()
                analysis = analyzer.analyze(question)
                if image_query_path and analysis.is_instruction:
                    search_query = None
                elif analysis.search_query:
                    search_query = analysis.search_query
             except:
                 pass

        input_image_context = "No input image provided."
        if image_query_path:
             try:
                from models.clip_model import get_clip_model
                clip = get_clip_model()
                candidates = ["chart", "diagram", "table", "screenshot", "photograph"] 
                label = clip.get_image_label(image_query_path, candidates)
                input_image_context = f"The user provided an image classified as: {label}."
             except:
                 pass

        # 1. Retrieve & Format
        results = self._retriever.retrieve(
            query=search_query if search_query else "",
            image_query_path=image_query_path
        )
        context = self._retriever.format_context(results)
        
        # 2. stream
        safe_question = question if question else "Analyze the provided input image context."
        prompt_value = self._prompt.format(
            context=context,
            input_image_context=input_image_context,
            question=safe_question,
        )
        
        for chunk in self._llm.stream(prompt_value):
            yield chunk
    
    def query_simple(self, question: str) -> str:
        """
        Simple query returning just the answer string.
        """
        response = self.query(question)
        return response.answer


def get_rag_chain() -> RAGChain:
    return RAGChain()
