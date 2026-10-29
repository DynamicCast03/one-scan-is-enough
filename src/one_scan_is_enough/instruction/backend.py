"""Generate instruction text through Chat Completions or Responses endpoints."""

import base64
import logging
import os
import time

from omegaconf import DictConfig, OmegaConf
from openai import DefaultHttpxClient, OpenAI

from one_scan_is_enough.instruction.prompts import SYSTEM_INSTRUCTION

logger = logging.getLogger(__name__)


class InstructionBackend:
    """Own one shared client and return text from labeled route and goal images.

    The configuration selects the API, transport and retry policy. Both APIs
    receive the same text and images in the same order.
    Use this object as a context manager to close its connection pool.
    """

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        backend = config.backend
        if backend.api not in ("chat_completions", "responses"):
            raise ValueError(f"Unknown instruction API: {backend.api}")
        if backend.max_retries <= 0:
            raise ValueError("Retry count must be positive.")
        api_key = os.environ.get(backend.api_key_env) if backend.api_key_env else "EMPTY"
        if api_key is None:
            raise RuntimeError(
                f"Set the API key in the environment variable {backend.api_key_env}, "
                "or set llm.backend.api_key_env=null for an endpoint without a key."
            )
        self.client = OpenAI(
            api_key=api_key,
            base_url=backend.base_url,
            timeout=backend.timeout_s,
            max_retries=0,
            http_client=DefaultHttpxClient(trust_env=backend.trust_env, proxy=backend.proxy),
        )

    def __enter__(self) -> "InstructionBackend":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.client.close()

    def complete(self, prompt: str, route: list[bytes], goal: list[bytes]) -> str:
        """Return one complete instruction, retrying failed or empty responses."""
        content = [{"type": "text", "text": prompt}]
        for prefix, frames in (("R", route), ("G", goal)):
            for index, frame in enumerate(frames, start=1):
                url = "data:image/jpeg;base64," + base64.b64encode(frame).decode("ascii")
                content += [
                    {"type": "text", "text": f"{prefix}{index}:"},
                    {"type": "image_url", "image_url": {"url": url}},
                ]
        backend = self.config.backend
        extra_body = OmegaConf.to_container(backend.extra_body, resolve=True)
        if backend.api == "responses":
            content = _parts(content)
        image_type = "input_image" if backend.api == "responses" else "image_url"
        count = sum(part["type"] == image_type for part in content)
        if backend.max_images is not None and count > backend.max_images:
            raise ValueError(f"Request has {count} images; endpoint limit is {backend.max_images}.")
        for attempt in range(backend.max_retries):
            try:
                if backend.api == "responses":
                    response = self.client.responses.create(
                        model=self.config.model,
                        instructions=SYSTEM_INSTRUCTION,
                        input=[{"role": "user", "content": content}],
                        store=False,
                        extra_body=extra_body,
                    )
                    if response.status != "completed":
                        raise RuntimeError(
                            f"Model response is {response.status}: {response.error or response.incomplete_details}"
                        )
                    result = response.output_text.strip()
                else:
                    response = self.client.chat.completions.create(
                        model=self.config.model,
                        messages=[
                            {"role": "system", "content": SYSTEM_INSTRUCTION},
                            {"role": "user", "content": content},
                        ],
                        extra_body=extra_body,
                    )
                    # Servers without a reasoning parser return thinking before </think>.
                    result = (
                        (response.choices[0].message.content or "")
                        .rpartition("</think>")[2]
                        .strip()
                    )
                if not result:
                    raise RuntimeError("The model returned an empty instruction.")
                return result
            except Exception as error:
                if attempt + 1 == backend.max_retries:
                    raise
                delay = min(60.0, 0.5 * 2**attempt)
                logger.warning("Model request failed (%s); retry in %.1f seconds", error, delay)
                time.sleep(delay)
        raise RuntimeError("Model request retries exhausted.")


def _parts(content: list[dict]) -> list[dict]:
    """Return the Responses input parts of Chat Completions message content."""
    parts = []
    for part in content:
        if part["type"] == "text":
            parts.append({"type": "input_text", "text": part["text"]})
        else:
            parts.append(
                {"type": "input_image", "image_url": part["image_url"]["url"], "detail": "auto"}
            )
    return parts
