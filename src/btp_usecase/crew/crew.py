from __future__ import annotations

from crewai import Crew, Process

from .agents import (
    make_api_agent,
    make_document_agent,
    make_input_agent,
    make_llm_agent,
    make_response_agent,
)
from .tasks import (
    make_external_api_task,
    make_finalize_response_task,
    make_llm_process_task,
    make_normalize_query_task,
    make_retrieve_document_task,
)


def build_crew(
    user_query: str = "",
    document_id: str | None = None,
    document_text: str | None = None,
) -> Crew:
    """
    Construct and return a CrewAI Crew that replicates the five-step LangGraph
    pipeline:

        input → document → llm → api → response

    The crew runs sequentially (Process.sequential) so each task has access to
    the outputs of all preceding tasks through the context list.
    """
    # ------------------------------------------------------------------
    # Agents
    # ------------------------------------------------------------------
    input_agent = make_input_agent()
    document_agent = make_document_agent()
    llm_agent = make_llm_agent()
    api_agent = make_api_agent()
    response_agent = make_response_agent()

    # ------------------------------------------------------------------
    # Tasks (sequential; each receives all prior tasks as context)
    # ------------------------------------------------------------------
    t_input = make_normalize_query_task(agent=input_agent, user_query=user_query)

    t_document = make_retrieve_document_task(
        agent=document_agent,
        document_id=document_id,
        document_text=document_text,
        context=[t_input],
    )

    t_llm = make_llm_process_task(
        agent=llm_agent,
        user_query=user_query,
        document_text=document_text,
        context=[t_input, t_document],
    )

    t_api = make_external_api_task(
        agent=api_agent,
        context=[t_input, t_document, t_llm],
    )

    t_response = make_finalize_response_task(
        agent=response_agent,
        context=[t_input, t_document, t_llm, t_api],
    )

    return Crew(
        agents=[input_agent, document_agent, llm_agent, api_agent, response_agent],
        tasks=[t_input, t_document, t_llm, t_api, t_response],
        process=Process.sequential,
        verbose=False,
    )


def run_crew(
    user_query: str,
    document_id: str | None = None,
    document_text: str | None = None,
) -> dict:
    """
    Build and kick-off the crew, then return a result dict that matches the
    shape previously returned by the LangGraph ``AgentService.run_agent``.
    """
    crew = build_crew(
        user_query=user_query,
        document_id=document_id,
        document_text=document_text,
    )
    crew_output = crew.kickoff()

    # CrewAI returns a CrewOutput object; the .raw attribute holds the final
    # task's string output, which is the response agent's output.
    final_response: str = getattr(crew_output, "raw", str(crew_output)) or "I could not generate a response."

    return {
        "document_id": document_id,
        "final_response": final_response,
        "messages": [user_query],
        "intermediate_results": {
            "llm": {
                "provider": "ollama",
                "model": None,   # model name resolved inside agents.py at build time
                "response": final_response,
            }
        },
    }
