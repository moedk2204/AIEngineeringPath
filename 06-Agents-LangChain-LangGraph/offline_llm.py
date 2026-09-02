"""
A stand-in chat model so every notebook in this module runs WITHOUT an API key.

It replays a fixed script of AIMessages instead of calling a provider. That is enough to
exercise the real LangChain and LangGraph machinery - the agent loop, tool execution,
state, routing and checkpointing all behave exactly as they would with a live model.

Use it to learn the mechanics. Swap in a real model (see `real_model()` below) when you
want real answers.

    from offline_llm import ScriptedChatModel, ai, tool_call

    llm = ScriptedChatModel(responses=[
        ai("", [tool_call("add", a=2, b=3)]),
        ai("2 + 3 = 5."),
    ])
"""

from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain.messages import AIMessage


class ScriptedChatModel(BaseChatModel):
    """Replays `responses` in order. The last one repeats if the agent loops further."""

    responses: List[AIMessage]
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-offline"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        message = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools, **kwargs):
        # A real provider validates the tool schemas here and tells the model about them.
        # The script already knows what it will call, so this is a no-op.
        return self


def ai(content: str = "", tool_calls: Optional[list] = None) -> AIMessage:
    """Build an AIMessage, optionally carrying tool calls."""
    return AIMessage(content=content, tool_calls=tool_calls or [])


def tool_call(name: str, call_id: Optional[str] = None, **args: Any) -> dict:
    """Build one tool call in the shape LangChain expects."""
    return {
        "name": name,
        "args": args,
        "id": call_id or f"call_{name}",
        "type": "tool_call",
    }


def real_model(model: str = "openai:gpt-4.1-mini", **kwargs):
    """
    Swap the scripted model for a real one.

    Requires the matching provider package and API key in your .env:

        openai:gpt-4.1-mini              pip install langchain-openai        OPENAI_API_KEY
        anthropic:claude-opus-5          pip install langchain-anthropic     ANTHROPIC_API_KEY
        google_genai:gemini-2.5-flash    pip install langchain-google-genai  GOOGLE_API_KEY
        ollama:llama3.2                  pip install langchain-ollama        (runs locally, free)

    Copy `.sample.env` to `.env` and fill in the key you have.
    """
    from dotenv import load_dotenv
    from langchain.chat_models import init_chat_model

    load_dotenv()
    return init_chat_model(model, **kwargs)
