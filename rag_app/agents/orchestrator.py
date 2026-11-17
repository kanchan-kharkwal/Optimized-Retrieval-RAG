"""Top-level routing, clarification, fan-out, and synthesis graph."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send, interrupt

from rag_app.config import Settings
from rag_app.models import AgentState, Decomposition, RouteDecision, SubQueryInput


def route_after_router(
    state: AgentState,
) -> Literal["clarify", "direct", "decompose"]:
    return {"clarify": "clarify", "direct": "direct"}.get(
        state["route"], "decompose"
    )


def build_graph(
    llm: BaseChatModel, retrieval_graph: Any, settings: Settings
) -> CompiledStateGraph[AgentState, Any, Any, Any]:
    def router_node(state: AgentState):
        count = state.get("clarification_count", 0)
        decider = llm.with_structured_output(RouteDecision)
        decision = decider.invoke(
            [
                SystemMessage(
                    content="You route user questions for a RAG system. Choose "
                    "'clarify' ONLY if the question is genuinely ambiguous "
                    "(missing referent, unclear scope) and you cannot "
                    "reasonably guess. Choose 'direct' for greetings or "
                    "general questions needing no documents. Otherwise "
                    "choose 'retrieve'."
                ),
                HumanMessage(content=state["question"]),
            ]
        )
        route = decision.route
        if route == "clarify" and count >= settings.max_clarifications:
            route = "retrieve"
        return {
            "route": route,
            "clarifying_question": decision.clarifying_question,
        }

    def clarify_node(state: AgentState):
        reply = interrupt(
            {
                "question": state.get("clarifying_question")
                or "Could you clarify what you mean?"
            }
        )
        return {
            "question": f"{state['question']}\n(User clarification: {reply})",
            "clarification_count": state.get("clarification_count", 0) + 1,
        }

    def direct_node(state: AgentState):
        return {"answer": llm.invoke(state["question"]).content}

    def decompose_node(state: AgentState):
        decomposer = llm.with_structured_output(Decomposition)
        result = decomposer.invoke(
            [
                SystemMessage(
                    content=f"Split the question into at most {settings.max_subqueries} "
                    "independent, self-contained sub-queries that can be "
                    "answered in parallel. Do not split single-part "
                    "questions."
                ),
                HumanMessage(content=state["question"]),
            ]
        )
        sub_queries = [
            query.strip() for query in result.sub_queries if query.strip()
        ][: settings.max_subqueries]
        return {"sub_queries": sub_queries or [state["question"]]}

    def fan_out(state: AgentState):
        return [
            Send("retrieval_agent", {"sub_query": query})
            for query in state["sub_queries"]
        ]

    def retrieval_agent_node(state: SubQueryInput):
        query = state["sub_query"]
        result = retrieval_graph.invoke(
            {"sub_query": query, "original_query": query, "attempts": 0}
        )
        return {
            "sub_answers": [
                {
                    "sub_query": query,
                    "answer": result["answer"],
                    "retries": result.get("attempts", 0),
                }
            ]
        }

    def synthesize_node(state: AgentState):
        findings = "\n\n".join(
            f"Sub-question: {item['sub_query']}\nFinding: {item['answer']}"
            for item in state["sub_answers"]
        )
        message = llm.invoke(
            [
                SystemMessage(
                    content="Combine the findings into one coherent answer to the "
                    "original question. Keep citations. Note any gaps "
                    "honestly; do not invent facts."
                ),
                HumanMessage(
                    content=f"Original question: {state['question']}\n\n{findings}"
                ),
            ]
        )
        return {"answer": message.content}

    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("direct", direct_node)
    graph.add_node("decompose", decompose_node)
    graph.add_node("retrieval_agent", retrieval_agent_node)
    graph.add_node("synthesize", synthesize_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "clarify": "clarify",
            "direct": "direct",
            "decompose": "decompose",
        },
    )
    graph.add_edge("clarify", "router")
    graph.add_conditional_edges("decompose", fan_out, ["retrieval_agent"])
    graph.add_edge("retrieval_agent", "synthesize")
    graph.add_edge("synthesize", END)
    graph.add_edge("direct", END)

    return graph.compile(checkpointer=MemorySaver())
