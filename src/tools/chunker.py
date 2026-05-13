"""
Text Chunking Utilities for MA-RAG Pipeline.

Provides sentence-based and section-based chunking for PubMed abstracts
to enable efficient processing by small language models.
"""

import re
from typing import List, Tuple
import logging
import os
import tempfile

logger = logging.getLogger(__name__)

try:
    import nltk
except ImportError:
    nltk = None


_NLTK_READY = False


def _bootstrap_nltk_tokenizers() -> None:
    """Ensure punkt tokenizer resources exist before sentence splitting starts.

    Downloads are routed to a writable temporary directory (default `/tmp`)
    to avoid PermissionError in read-only serverless environments. The
    path can be overridden with the `MRAGE_NLTK_DOWNLOAD_DIR` env var.
    """
    global _NLTK_READY

    if nltk is None or _NLTK_READY:
        return

    # Prefer explicit /tmp for serverless containers, but fall back to
    # the platform tempdir if /tmp is unavailable (Windows, etc.). Allow
    # override via env var for testing.
    download_dir = os.getenv("MRAGE_NLTK_DOWNLOAD_DIR", "/tmp")
    if not os.path.isdir(download_dir):
        download_dir = tempfile.gettempdir()

    for resource_name, download_name in (
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("tokenizers/punkt", "punkt"),
    ):
        try:
            nltk.data.find(resource_name)
        except LookupError:
            try:
                # Patch: call download() without download_dir if not supported (FakeNLTK in tests)
                import inspect
                download_fn = getattr(nltk, "download", None)
                if download_fn:
                    params = inspect.signature(download_fn).parameters
                    if "download_dir" in params:
                        nltk.download(download_name, download_dir=download_dir, quiet=True)
                    else:
                        nltk.download(download_name, quiet=True)
                if download_dir not in nltk.data.path:
                    nltk.data.path.insert(0, download_dir)
            except Exception as e:
                logger.warning("NLTK download failed for %s to %s: %s", download_name, download_dir, e)

    _NLTK_READY = True


_bootstrap_nltk_tokenizers()


class TextChunker:
    """
    Chunk text into smaller pieces for efficient LLM processing.
    
    Strategies:
    - sentence: Split by sentences with overlap
    - section: Split by abstract sections (Background/Methods/Results/Conclusion)
    - fixed: Fixed character count chunks
    """
    
    def __init__(
        self, 
        strategy: str = "sentence",
        sentences_per_chunk: int = 6,
        overlap_sentences: int = 2,
        max_chunk_chars: int = 500
    ):
        """
        Initialize chunker.
        
        Args:
            strategy: "sentence", "section", or "fixed"
            sentences_per_chunk: Number of sentences per chunk (for sentence strategy)
            overlap_sentences: Overlap between chunks (for sentence strategy)
            max_chunk_chars: Maximum characters per chunk (for fixed strategy)
        """
        self.strategy = strategy
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap_sentences = overlap_sentences
        self.max_chunk_chars = max_chunk_chars
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        try:
            # Try using nltk's sentence tokenizer if available
            if nltk is not None:
                try:
                    sentences = nltk.sent_tokenize(text)
                    return [s.strip() for s in sentences if s.strip()]
                except LookupError:
                    # nltk data not downloaded, use fallback approach
                    pass
        except Exception:
            pass

        protected = text
        protected_tokens: dict[str, str] = {}

        def _protect_token(original: str) -> str:
            token = f"<ABBR_TOKEN_{len(protected_tokens)}>"
            protected_tokens[token] = original
            return token

        def _protect_numbered_reference(match: re.Match) -> str:
            label = match.group(1)
            number = match.group(2)
            return _protect_token(f"{label}. {number}")

        protected = re.sub(r"\b(Fig|Figs|Table|Eq|Ref|Refs)\.\s*(\d+)", _protect_numbered_reference, protected, flags=re.IGNORECASE)

        abbreviation_patterns = {
            r"\b(e\.g|i\.e)\.": "<ABBR_EG_IE>",
            r"\b(et al)\.": "<ABBR_ETAL>",
            r"\b(Fig|Figs|Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|No|Eq|Ref|Refs|St)\.": lambda m: f"<ABBR_{m.group(1).upper()}>",
        }

        for pattern, replacement in abbreviation_patterns.items():
            protected = re.sub(pattern, replacement, protected, flags=re.IGNORECASE)

        protected = re.sub(r"\b([A-Z])\.\s+(?=\d)", r"\1<ABBR_INITIAL> ", protected)

        sentence_boundary = r"(?<=[.!?])\s+(?=[\"'“”‘’(\[]*[A-Z0-9])"
        sentences = re.split(sentence_boundary, protected)
        sentences = [
            s.replace("<ABBR_EG_IE>", "e.g.")
             .replace("<ABBR_ETAL>", "et al.")
             .replace("<ABBR_INITIAL>", ".")
             .replace("<ABBR_FIG>", "Fig.")
             .replace("<ABBR_FIGS>", "Figs.")
             .replace("<ABBR_DR>", "Dr.")
             .replace("<ABBR_MR>", "Mr.")
             .replace("<ABBR_MRS>", "Mrs.")
             .replace("<ABBR_MS>", "Ms.")
             .replace("<ABBR_PROF>", "Prof.")
             .replace("<ABBR_SR>", "Sr.")
             .replace("<ABBR_JR>", "Jr.")
             .replace("<ABBR_VS>", "vs.")
             .replace("<ABBR_NO>", "No.")
             .replace("<ABBR_EQ>", "Eq.")
             .replace("<ABBR_REF>", "Ref.")
             .replace("<ABBR_REFS>", "Refs.")
             .replace("<ABBR_ST>", "St.")
             .strip()
            for s in sentences
            if s and s.strip()
        ]

        if protected_tokens:
            restored_sentences = []
            for sentence in sentences:
                restored = sentence
                for token, original in protected_tokens.items():
                    restored = restored.replace(token, original)
                restored_sentences.append(restored)
            sentences = restored_sentences

        return sentences if sentences else [text]
    
    def _chunk_by_sentences(self, text: str) -> List[str]:
        """Chunk text using sliding window over sentences."""
        sentences = self._split_sentences(text)
        
        if len(sentences) <= self.sentences_per_chunk:
            return [text] if text.strip() else []
        
        chunks = []
        step = self.sentences_per_chunk - self.overlap_sentences
        step = max(1, step)  # Ensure at least 1 sentence step
        
        for i in range(0, len(sentences), step):
            chunk_sentences = sentences[i:i + self.sentences_per_chunk]
            if chunk_sentences:
                chunk = '. '.join(chunk_sentences)
                if not chunk.endswith('.'):
                    chunk += '.'
                chunks.append(chunk)
            
            # Stop if we've covered all sentences
            if i + self.sentences_per_chunk >= len(sentences):
                break
        
        return chunks
    
    def _chunk_by_section(self, text: str) -> List[str]:
        """
        Chunk by detecting abstract sections.
        Common patterns: BACKGROUND:, METHODS:, RESULTS:, CONCLUSION:
        """
        # Section header patterns
        section_patterns = [
            r'\b(BACKGROUND|Background|INTRODUCTION|Introduction):?\s*',
            r'\b(METHODS?|Methods?|MATERIALS?\s+AND\s+METHODS?):?\s*',
            r'\b(RESULTS?|Results?|FINDINGS?|Findings?):?\s*',
            r'\b(CONCLUSIONS?|Conclusions?|DISCUSSION|Discussion):?\s*',
        ]
        
        # Try to find sections
        combined_pattern = '|'.join(section_patterns)
        parts = re.split(f'({combined_pattern})', text, flags=re.IGNORECASE)
        
        if len(parts) <= 1:
            # No sections found, fall back to sentence chunking
            return self._chunk_by_sentences(text)
        
        chunks = []
        current_chunk = ""
        
        for part in parts:
            if part and re.match(combined_pattern, part, re.IGNORECASE):
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = part.strip() + " "
            elif part and part.strip():
                current_chunk += part.strip() + " "
        
        # Add last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]
    
    def _chunk_by_fixed_size(self, text: str) -> List[str]:
        """Chunk by fixed character count with word boundary awareness."""
        if len(text) <= self.max_chunk_chars:
            return [text] if text.strip() else []
        
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_len = len(word) + 1  # +1 for space
            if current_length + word_len > self.max_chunk_chars and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += word_len
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def chunk(self, text: str) -> List[str]:
        """
        Chunk text using the configured strategy.
        
        Args:
            text: Input text to chunk.
            
        Returns:
            List of text chunks.
        """
        if not text or not text.strip():
            return []
        
        if self.strategy == "sentence":
            return self._chunk_by_sentences(text)
        elif self.strategy == "section":
            return self._chunk_by_section(text)
        elif self.strategy == "fixed":
            return self._chunk_by_fixed_size(text)
        else:
            logger.warning(f"Unknown strategy '{self.strategy}', using sentence")
            return self._chunk_by_sentences(text)
    
    def chunk_documents(self, documents: List[str]) -> List[Tuple[int, str]]:
        """
        Chunk multiple documents, preserving document index.
        
        Args:
            documents: List of document texts.
            
        Returns:
            List of (doc_index, chunk_text) tuples.
        """
        all_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            chunks = self.chunk(doc)
            for chunk in chunks:
                all_chunks.append((doc_idx, chunk))
        
        logger.info(f"Chunked {len(documents)} docs into {len(all_chunks)} chunks")
        return all_chunks

    def chunk_documents_with_metadata(self, documents: List[str]) -> List[dict]:
        """
        Chunk multiple documents and preserve metadata for each chunk.

        Returns:
            List of dictionaries containing chunk_id, doc_index, chunk_index,
            sentence_count, and chunk_text.
        """
        all_chunks: List[dict] = []
        chunk_id = 0

        for doc_idx, doc in enumerate(documents):
            chunks = self.chunk(doc)
            for chunk_index, chunk in enumerate(chunks):
                sentence_count = len(self._split_sentences(chunk))
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "doc_index": doc_idx,
                    "chunk_index": chunk_index,
                    "sentence_count": sentence_count,
                    "chunk_text": chunk,
                })
                chunk_id += 1

        logger.info(f"Chunked {len(documents)} docs into {len(all_chunks)} metadata chunks")
        return all_chunks
    
    def batch_chunks(
        self, 
        chunks: List[Tuple[int, str]], 
        batch_size: int = 3
    ) -> List[List[Tuple[int, str]]]:
        """
        Group chunks into batches for processing.
        
        Args:
            chunks: List of (doc_index, chunk_text) tuples.
            batch_size: Number of chunks per batch.
            
        Returns:
            List of batches, each batch is a list of tuples.
        """
        batches = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batches.append(batch)
        
        logger.info(f"Created {len(batches)} batches of {batch_size} chunks each")
        return batches


# Convenience function
def chunk_abstracts(
    abstracts: List[str], 
    sentences_per_chunk: int = 8,
    overlap: int = 2
) -> List[Tuple[int, str]]:
    """
    Convenience function to chunk PubMed abstracts.
    
    Args:
        abstracts: List of abstract texts.
        sentences_per_chunk: Sentences per chunk.
        overlap: Sentence overlap between chunks.
        
    Returns:
        List of (abstract_index, chunk_text) tuples.
    """
    chunker = TextChunker(
        strategy="sentence",
        sentences_per_chunk=sentences_per_chunk,
        overlap_sentences=overlap
    )
    return chunker.chunk_documents(abstracts)
