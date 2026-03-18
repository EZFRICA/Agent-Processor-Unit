from .letta_cloud_client import letta
from logger import get_logger

logger = get_logger(__name__)


from typing import Optional

def get_core_block_content(agent_id: str, label: str) -> Optional[str]:
    """Fetch the content of a core memory block from Letta Cloud API. Returns None if 404 (Missing)."""
    try:
        block = letta.agents.blocks.retrieve(label, agent_id=agent_id)
        if block.value is not None:
             return block.value
    except Exception as e:
        if "404" in str(e):
             return None
        logger.warning("Could not read block '%s' from Letta API: %s", label, e)
    return ""


def compile_working_context(agent_id: str, relevant_blocks: list[dict], query: str = "") -> str:
    """
    Assemble the final Working Context string to inject into the Gemini prompt.
    Only the active blocks selected by the DLL are included.
    """
    context_parts = []

    for block in relevant_blocks:
        label = block["id"]
        content = get_core_block_content(agent_id, label)
        if content is None:
            content = ""
            
        context_parts.append(f"--- BLOCK: {block['label'].upper()} ({block['type']}) ---")
        context_parts.append(content)
        context_parts.append("")  # blank line separator

    return "\n".join(context_parts)
