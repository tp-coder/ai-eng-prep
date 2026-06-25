import logging
import time
import json

from dataclasses import dataclass
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from app.config import Settings, get_settings
from app.schemas import AssistantResponse
from app.tools import SEARCH_DOCS_TOOL, execute_tool


logger = logging.getLogger(__name__)


class LLMConfigurationError(RuntimeError):
    pass


class LLMResponseParsingError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMResponse:
    parsed: AssistantResponse
    model: str
    latency_ms: int


SYSTEM_PROMPT = """
You are an AI engineering assistant.
Your job is to answer clearly and pragmatically.
Rules:
- Be direct.
- Do not invent missing facts.
- If the request lacks context, say what was missing.
- Use source_references only when  explicit source materials are provided.
- Keep next_actions practical and short.
""".strip()


def build_prompt(
    user_prompt: str,
    retrieved_context: str | None = None,
) -> str:

    if not retrieved_context:
        return user_prompt

    return f"""
    User question: {user_prompt}
    Retrieved context: {retrieved_context}
    Instructions:
    - Answer the user's question using only the retrieved context.
    - If context is insufficient, say it clearly.
    - Include the source labels used in source_references.
    """.strip()


AGENT_SYSTEM_PROMPT = """
You are an AI engineering assistant with a document search tool (search_docs).
- Call search_docs when answering needs facts that might exist in the indexed documents.
- For general questions you can answer directly, do NOT call the tool.
- When you use retrieved context, ground the answer in it and put the source labels in source_references.
- If the tool returns nothing relevant and you lack the facts, say what's missing rather than guessing.
""".strip()


MAX_TOOL_ITERATIONS = 5


def complete_with_tools(self, prompt: str) -> LLMResponse:
    started_at = time.perf_counter()
    input_items = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    tool_calls_made = 0
    for _ in range(MAX_TOOL_ITERATIONS):
        response = self.client.responses.parse(
            model=self.settings.openai_model,
            input=input_items,
            tools=[SEARCH_DOCS_TOOL],
            text_format=AssistantResponse,
        )

        function_calls = [
            item for item in response.output if item.type == "function_call"]

        if not function_calls:  # no more tool calls. model is done → final answer
            parsed = response.output_parsed
            if parsed is None:
                raise LLMResponseParsingError(
                    "No parsed AssistantResponse and no tool call.")

            latency_ms = int((time.perf_counter() - started_at) * 1000)
            logger.info("agent_completed tool_calls=%s latency_ms=%s",
                        tool_calls_made, latency_ms)
            return LLMResponse(
                parsed=parsed,
                model=self.settings.openai_model,
                latency_ms=latency_ms,
            )

        input_items += response.output
        for call in function_calls:
            result = execute_tool(call.name, json.loads(call.arguments))
            tool_calls_made += 1
            input_items.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": result,
            })

    raise RuntimeError(
        f"Agent exceeded MAX_TOOL_ITERATIONS={MAX_TOOL_ITERATIONS}")


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

        if not self.settings.openai_api_key:
            raise LLMConfigurationError(
                "OPENAI_API_KEY is missing. Add it to your .env file before trying again."
            )

        self.client = OpenAI(
            api_key=self.settings.openai_api_key,
            timeout=self.settings.openai_timeout_seconds,
            max_retries=self.settings.openai_max_retries,
        )

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
        reraise=True
    )
    def complete(self, prompt: str, retrieved_context: str | None = None) -> LLMResponse:
        started_at = time.perf_counter()
        final_prompt = build_prompt(prompt, retrieved_context)

        logger.info(
            "llm_request_started model=%s prompt_length=%s has_retrieved_context=%s",
            self.settings.openai_model,
            len(final_prompt),
            retrieved_context is not None,
        )

        try:
            response = self.client.responses.parse(
                model=self.settings.openai_model,
                input=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": final_prompt,
                    }
                ],
                text_format=AssistantResponse,
            )
        except Exception:
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            logger.exception(
                "llm_request_failed model=%s latency_ms=%s",
                self.settings.openai_model,
                latency_ms,
            )
            raise

        latency_ms = int((time.perf_counter() - started_at) * 1000)

        if response.output_parsed is None:
            logger.error(
                "llm_response_parse_failed model=%s latency_ms=%s",
                self.settings.openai_model,
                latency_ms,
            )
            raise LLMResponseParsingError(
                "Model response could not be parsed into AssistantResponse."
            )

        logger.info(
            "llm_request_completed model=%s latency_ms=%s confidence=%s missing_context_count=%s next_actions_count=%s",
            self.settings.openai_model,
            latency_ms,
            response.output_parsed.confidence,
            len(response.output_parsed.missing_context),
            len(response.output_parsed.next_actions),
        )

        return LLMResponse(
            parsed=response.output_parsed,
            model=self.settings.openai_model,
            latency_ms=latency_ms,
        )
