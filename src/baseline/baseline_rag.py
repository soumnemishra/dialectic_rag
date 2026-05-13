import logging
from typing import List
from src.pubmed_client import PubMedClient
from src.core.registry import ModelRegistry
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

class BaselineRAG:
    """Standard RAG baseline (no epistemic reasoning)."""
    
    def __init__(self):
        self.pubmed = PubMedClient()
        self.llm = ModelRegistry.get_flash_llm(temperature=0.0)

    async def answer(self, question: str) -> str:
        """Retrieve top 5 abstracts and generate an answer."""
        # 1. Retrieve
        pmids, _ = await self.pubmed.esearch(question, max_results=5)
        if not pmids:
            return "No evidence found."
        
        articles = await self.pubmed.efetch(pmids)
        context = "\n\n".join([f"PMID {a.pmid}: {a.abstract}" for a in articles])
        
        # 2. Generate
        prompt = ChatPromptTemplate.from_template("""
        Answer the clinical question using the provided context.
        Question: {question}
        Context: {context}
        """)
        
        try:
            res = await self.llm.ainvoke(prompt.format(question=question, context=context))
            return res.content
        except Exception as e:
            logger.error(f"Baseline RAG failed: {e}")
            return "Error generating answer."
