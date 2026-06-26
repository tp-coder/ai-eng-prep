import json
import logging
import psycopg


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS traces (
    id              BIGSERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    mode            TEXT NOT NULL, -- 'rag' | 'agent' -- this is the comparison key
    prompt          TEXT NOT NULL,
    model           TEXT NOT NULL,
    tools_used      TEXT[],
    model_calls     INT NOT NULL,
    input_tokens    INT NOT NULL,
    output_tokens   INT NOT NULL,
    cost_usd        NUMERIC(12,6) NOT NULL,
    latency_ms      INT NOT NULL,
    answer          TEXT,
    trajectory      JSONB
);
"""

logger = logging.getLogger(__name__)


class TraceStore:
    def __init__(self, db_url: str) -> None:
        self.db_url = db_url
        with psycopg.connect(self.db_url) as conn:
            conn.execute(CREATE_TABLE_SQL)

    def save(self, trace: dict) -> None:
        try:
            with psycopg.connect(self.db_url) as conn:
                conn.execute(
                    """INSERT INTO traces
                    (mode, prompt, model, tools_used, model_calls, input_tokens,
                    output_tokens, cost_usd, latency_ms, answer, trajectory)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (trace["mode"], trace["prompt"], trace["model"], trace["tools_used"], trace["model_calls"], trace["input_tokens"],
                     trace["output_tokens"], trace["cost_usd"], trace["latency_ms"], trace["answer"], json.dumps(trace.get("trajectory"))),
                )
                conn.commit()
        except Exception as error:
            logger.warning("trace_error_save_failed error=%s", error)
