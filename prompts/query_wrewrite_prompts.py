
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Prompt for query rewriting (to improve retrieval)
QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query optimization assistant. Your task is to rewrite user queries to improve document retrieval.

## Instructions:
1. Expand abbreviations and acronyms
2. Add relevant synonyms or related terms
3. Make implicit context explicit
4. Keep the core intent of the original query
5. Output ONLY the rewritten query, nothing else"""),
    ("human", "Original query: {query}\n\nRewritten query:"),
])


# Prompt for generating document summaries
SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a document summarization assistant. Create concise, informative summaries.

## Instructions:
1. Capture the main points and key information
2. Maintain factual accuracy
3. Keep summaries under 200 words
4. Use clear, professional language"""),
    ("human", """Please summarize the following document:

{document}

Summary:"""),
])


def get_query_rewrite_prompt() -> ChatPromptTemplate:
    """Get the query rewriting prompt template."""
    return QUERY_REWRITE_PROMPT


def get_summary_prompt() -> ChatPromptTemplate:
    """Get the document summary prompt template."""
    return SUMMARY_PROMPT
