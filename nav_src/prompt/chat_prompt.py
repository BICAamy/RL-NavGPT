"""Backend-neutral chat messages shared by inference and RL training."""

from typing import Dict, List


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant for embodied navigation. Follow the format "
    "requested in the user prompt exactly."
)


def build_chat_messages(prompt: str) -> List[Dict[str, str]]:
    """Return the messages that HF will render with ``apply_chat_template``."""

    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
