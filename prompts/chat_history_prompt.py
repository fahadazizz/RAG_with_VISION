from langchain_core.prompts import PromptTemplate


CHAT_HISTORY_PROMPT = PromptTemplate.from_template(
            """Given the following conversation history, can you answer the user's question?
            If yes, provide the answer AND the original sources mentioned in previous turns.
            If the history does not contain enough information to answer partially or fully, reply exactly "NO_MEMORY_CONTEXT".
            
            History:
            {history}
            
            Question: {question}
            
            Answer:"""
)


def chat_history_prompt() -> PromptTemplate:
    return CHAT_HISTORY_PROMPT