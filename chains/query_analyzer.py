from typing import Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from models.llm import get_ollama_llm

class QueryAnalysisResult(BaseModel):
    """Result of analyzing the user's query."""
    is_instruction: bool = Field(description="True if the text is primarily an instruction about the image (e.g., 'describe this', 'analyze this'). False if it contains specific search keywords.")
    search_query: Optional[str] = Field(description="Extracted search query if applicable. None if it is purely an instruction.")

class QueryAnalyzer:
    def __init__(self):
        self.llm = get_ollama_llm()._llm
        self.parser = JsonOutputParser(pydantic_object=QueryAnalysisResult)
        
        system_prompt = """You are a Query Analyzer for a Multimodal RAG system.
Your task is to analyze the user's text input and determine if it should be used for retrieving documents or if it is merely an instruction for the AI (e.g., "describe this image").

- If the user provides an image + "Analyze this image", the text is an INSTRUCTION. Search Query should be null.
- If the user provides an image + "Find similar charts", the text is a SEARCH QUERY. Search Query = "charts".
- If the user provides text only "What is the capital of France?", it is a SEARCH QUERY.

Output JSON: {{"is_instruction": bool, "search_query": string | null}}"""

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        self.chain = self.prompt | self.llm | self.parser

    def analyze(self, query: str) -> QueryAnalysisResult:
        try:
            result = self.chain.invoke({"input": query})
            return QueryAnalysisResult(**result)
        except Exception as e:
            print(f"Query Analysis Failed: {e}")
            # Fallback: Assume it's a search query if we can't parse
            return QueryAnalysisResult(is_instruction=False, search_query=query)

def get_query_analyzer() -> QueryAnalyzer:
    return QueryAnalyzer()
