# FILE: src/exceptions.py
"""
Custom exception hierarchy for the Medical RAG Chatbot.

Provides clear separation between transient (retryable) and permanent errors.

Example Usage:
    from src.exceptions import TransientError, PermanentError
    
    try:
        response = fetch_pubmed_data(query)
    except TransientError as e:
        # Retry with backoff
        pass
    except PermanentError as e:
        # Log and fail
        pass
"""


class IngestError(Exception):
    """Base exception for all ingestion/processing errors."""
    
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class TransientError(IngestError):
    """
    Transient error that may succeed on retry.
    
    Examples: Network timeouts, rate limits, temporary service unavailability.
    """
    pass


class PermanentError(IngestError):
    """
    Permanent error that will not succeed on retry.
    
    Examples: Invalid API key, malformed query, resource not found.
    """
    pass


class PubMedAPIError(TransientError):
    """Error from PubMed API calls."""
    pass
