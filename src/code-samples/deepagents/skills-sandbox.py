"""Deep agent + skills demo: sync skills from store (server) or load from disk (local), then run."""

import os
import re
from pathlib import Path

# :snippet-start: skills-sandbox-py
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, LocalShellBackend, StoreBackend
from daytona import Daytona
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_daytona import DaytonaSandbox
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore

# :remove-start:
_PROJECT_ROOT = Path(__file__).resolve().parent
_TIME_TXT = _PROJECT_ROOT / "time.txt"
# UTC line written by skills/write_current_time/scripts/write_time.sh
_TIME_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

# :remove-end:

def _safe_filename(key: str) -> str:
    name = key.split("/")[-1]
    if ".." in name or any(c in name for c in ("*", "?")):
        raise ValueError(f"Invalid key: {key}")
    return name


class SkillSyncMiddleware(AgentMiddleware):
    """Upload skill scripts from the store into the sandbox before each run."""

    def __init__(self, backend: CompositeBackend):
        super().__init__()
        self.backend = backend

    async def abefore_agent(self, state: AgentState, runtime: Runtime) -> None:
        if runtime.store is None or runtime.server_info is None or runtime.server_info.user is None:
            return None
        user_id = runtime.server_info.user.identity
        store = runtime.store
        files = []
        for item in await store.asearch(("skills", user_id)):
            name = _safe_filename(item.key)
            files.append((f"/skills/{name}", item.value["content"].encode()))
        if files:
            await self.backend.aupload_files(files)
        return None


backend = CompositeBackend(
    sandbox = Daytona().create()
    default=DaytonaSandbox(sandbox=sandbox),
    routes={
        "/skills/": StoreBackend(
            namespace=lambda rt: ("skills", rt.server_info.user.identity),  # [!code highlight]
        ),
    },
)

agent = create_deep_agent(
    model="claude-haiku-4-5",
    backend=backend,
    skills=["/skills/"],
    checkpointer=InMemorySaver(),
)

# :snippet-end:

def verify_time_txt(path: Path = _TIME_TXT) -> str:
    """Assert `time.txt` exists and matches the write-current-time skill output format."""
    if not path.is_file():
        msg = f"Expected {path} to exist after running the skill."
        raise AssertionError(msg)
    line = path.read_text(encoding="utf-8").strip()
    if not _TIME_PATTERN.fullmatch(line):
        msg = f"Expected one ISO-8601 UTC line in {path}, got {line!r}"
        raise AssertionError(msg)
    return line


def main() -> None:
    """Run the agent with a local shell backend; requires LLM credentials (e.g. ANTHROPIC_API_KEY)."""
    if _TIME_TXT.exists():
        _TIME_TXT.unlink()

    prompt = (
        "Use the write_current_time skill: read its SKILL.md under /skills/, "
        "then run the shell script it describes so that a file named time.txt is created in the "
        "workspace root with the current UTC time. Use the execute tool to run bash."
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config={"configurable": {"thread_id": "skills-sandbox-demo"}},
    )
    messages = result.get("messages") or []
    if messages:
        print(messages[-1].content)

    line = verify_time_txt()
    print(f"OK: {_TIME_TXT} contains {line!r}")


if __name__ == "__main__":
    main()
