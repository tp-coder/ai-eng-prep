# Structured Output Notes

The project uses Pydantic schemas to make LLM responses predictable.

Instead of accepting arbitrary free-form text, the app expects an AssistantResponse object with these fields:

- answer
- confidence
- missing_context
- next_actions
- source_references

This makes the assistant easier to test, display, and integrate with backend workflows.

Structured output is important because AI applications often fail when downstream code assumes that model output follows a format but the model returns something unexpected.

The current schema is intentionally simple. Later phases may add specialized schemas for retrieved answers, generated tickets, evaluation results, and tool calls.