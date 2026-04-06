import os
from utils.config import LANGSMITH_PROJECT, get_langsmith_key


def setup_langsmith():
    key = get_langsmith_key()
    if not key:
        return False
    os.environ["LANGCHAIN_API_KEY"]    = key
    os.environ["LANGSMITH_API_KEY"]    = key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"]    = LANGSMITH_PROJECT
    return True


def get_recent_traces(limit=20):
    key = get_langsmith_key()
    if not key:
        return []
    try:
        from langsmith import Client
        client = Client(api_key=key)
        runs = list(client.list_runs(
            project_name=LANGSMITH_PROJECT,
            limit=limit,
            is_root=True,
        ))
        result = []
        for r in runs:
            latency = None
            if r.end_time and r.start_time:
                latency = round((r.end_time - r.start_time).total_seconds() * 1000)
            result.append({
                "id":         str(r.id)[:8],
                "name":       r.name or "—",
                "status":     "OK" if r.error is None else "ERROR",
                "latency_ms": latency,
            })
        return result
    except Exception:
        return []
