set -euo pipefail

questions=(
    "What does phase 3 of the ai-eng-prep project add?"
    "What is the long-term goal described for the ai-eng-prep project?"
    "What are the five fields of the AssistantResponse schema in this project?"
    "What does phase 4 of the project add?"
    "What is 15 percent of 240?"
    "What is 47 times 89?"
    "What is the capital of Australia?"
    "Who wrote the play Hamlet?",
)

cd "$(dirname "$0")/.."

for question in "${questions[@]}"; do
    uv run python -m app.main "$question" --allow-llm-general
    uv run python -m app.main "$question" --agent
done