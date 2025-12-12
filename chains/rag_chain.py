from typing import Optional, Generator, List
from dataclasses import dataclass

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.memory import ConversationSummaryBufferMemory

from chains.retriever import get_retriever
from models.llm import get_ollama_llm
from prompts.rag_prompts import get_rag_prompt
from prompts.chat_history_prompt import chat_history_prompt


@dataclass
class RAGResponse:
    answer: str
    sources: List[dict]
    query: str
    
    def __str__(self) -> str:
        return self.answer


class RAGChain:
    
    def __init__(self):
        self._retriever = get_retriever()
        self._llm = get_ollama_llm()
        self._prompt = get_rag_prompt()
        
        self.memory = ConversationSummaryBufferMemory(
            llm=self._llm._llm,
            max_token_limit=2000,
            memory_key="chat_history",
            return_messages=True,
        )

        self._chain = self._build_chain()

        

    def _build_chain(self):
        """
        Build the LCEL chain for RAG.
        
        Returns:
            LCEL chain = prompted LLM
        """
        # We pass "context" and "question" directly to invoke()
        chain = (
            self._prompt
            | self._llm._llm
            | StrOutputParser()
        )
        
        return chain


    def _consult_memory(self, question: str) -> Optional[str]:
        """
        Check if the question can be answered from memory.
        """
        history = self.memory.load_memory_variables({}).get("chat_history", [])
        if not history:
            return None
            
        check_prompt = chat_history_prompt()
        
        chain = check_prompt | self._llm._llm | StrOutputParser()
        response = chain.invoke({"history": history, "question": question})
        
        if "NO_MEMORY_CONTEXT" in response:
            return None
        
        return response


    def query(
        self,
        question: str,
        image_query_path: Optional[str] = None,
    ) -> RAGResponse:
        """
        Execute a RAG query and get a response.
        
        Args:
            question: User question
            image_query_path: Optional path to query image for multimodal search
            
        Returns:
            RAGResponse with answer and sources
        """

        print("Checking memory...")
        memory_answer = self._consult_memory(question)
        
        if memory_answer:
            print("Answer found in memory!")
            self.memory.save_context({"input": question}, {"output": memory_answer})
            
            return RAGResponse(
                answer=memory_answer,
                sources=[{"source": "Conversation Memory"}],
                query=question,
            )

        print("Retrieving from Vector DB...")

        # 1. Retrieve documents (Multimodal if image provided)
        results = self._retriever.retrieve(
            query=question,
            image_query_path=image_query_path
        )
        
        # 2. Format context
        context = self._retriever.format_context(results)
        
        # 3. Extract sources
        sources = [result.metadata for result in results]
        
        # 4. Generate answer
        response = self._chain.invoke({
            "context": context,
            "question": question
        })
        
         # 6. Save to Memory (WITH SOURCES)
        source_strings = [s.get('filename', 'Unknown') for s in sources]
        full_response_to_store = f"{response}\n\nSources: {', '.join(source_strings)}"
        
        self.memory.save_context({"input": question}, {"output": full_response_to_store})
        
        
        return RAGResponse(
            answer=response,
            sources=sources,
            query=question,
        )
    
    def stream_query(
        self,
        question: str,
        image_query_path: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Stream a RAG query response.
        
        Args:
            question: User question
            image_query_path: Optional path to query image for multimodal search
            
        Yields:
            Response chunks
        """
        # 1. Retrieve & Format
        results = self._retriever.retrieve(
            query=question,
            image_query_path=image_query_path
        )
        context = self._retriever.format_context(results)
        
        # 2. stream
        prompt_value = self._prompt.format(
            context=context,
            question=question,
        )
        
        for chunk in self._llm.stream(prompt_value):
            yield chunk
    
    def query_simple(self, question: str) -> str:
        """
        Simple query returning just the answer string.
        
        Args:
            question: User question
            
        Returns:
            Answer string
        """
        response = self.query(question)
        return response.answer


def get_rag_chain() -> RAGChain:
    return RAGChain()
