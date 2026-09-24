from __future__ import annotations

from crewai import Agent

from ..llm.ollama_provider import OllamaProvider
from ..core.config import get_settings


def _ollama_llm():
    """Return a plain string LLM identifier accepted by CrewAI's Ollama integration."""
    settings = get_settings()
    # CrewAI accepts "ollama/<model>" as the llm parameter when using Ollama.
    return f"ollama/{settings.llm_model}"


def make_input_agent() -> Agent:
    """Normalises and validates the incoming user query."""
    return Agent(
        role="Query Normalizer",
        goal="Clean, trim and validate the user query so downstream agents receive a well-formed prompt.",
        backstory=(
            "You are a pre-processing specialist responsible for sanitising raw user input "
            "before it reaches any retrieval or inference step."
        ),
        llm=_ollama_llm(),
        verbose=False,
        allow_delegation=False,
    )


def make_document_agent() -> Agent:
    """Retrieves and surfaces document context for the LLM agent."""
    return Agent(
        role="Document Retriever",
        goal="Extract and return the relevant document text that should be used as context.",
        backstory=(
            "You are a document specialist. Given a document identifier and its raw text, "
            "you surface exactly the content the answering agent needs."
        ),
        llm=_ollama_llm(),
        verbose=False,
        allow_delegation=False,
    )


def make_llm_agent() -> Agent:
    """Generates an answer from document context using the configured LLM."""
    return Agent(
        role="LLM Analyst",
        goal="Answer the user query using only the provided document context.",
        backstory=(
            "You are a privacy-aware language model analyst. You read unstructured document "
            "content and produce accurate, concise answers while avoiding hallucination."
        ),
        llm=_ollama_llm(),
        verbose=False,
        allow_delegation=False,
    )


def make_api_agent() -> Agent:
    """Simulates an external API call and attaches enrichment metadata."""
    return Agent(
        role="External API Integrator",
        goal="Call the external API endpoint and attach the result metadata to the workflow.",
        backstory=(
            "You are responsible for any side-effects that require external service calls, "
            "such as logging, enrichment, or data exfiltration checks."
        ),
        llm=_ollama_llm(),
        verbose=False,
        allow_delegation=False,
    )


def make_response_agent() -> Agent:
    """Finalises and formats the response that will be returned to the caller."""
    return Agent(
        role="Response Formatter",
        goal="Produce a clean, well-formed final response from the analysis results.",
        backstory=(
            "You are the last stage in the pipeline. You take intermediate results and "
            "synthesise them into a polished, user-facing answer."
        ),
        llm=_ollama_llm(),
        verbose=False,
        allow_delegation=False,
    )
