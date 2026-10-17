"""One wrapper per API provider (OpenAI, Anthropic, Google): send one prompt, get one answer; count tokens.

Every answer comes back in the same shape, so the rest of the pipeline never needs to know which SDK produced it.
API keys are read from the environment: OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY (see .env.example).
"""

from functools import cache

# Retries for rate limits (429), server errors (5xx) and dropped connections, with exponential backoff.
MAX_RETRIES = 8

PROVIDERS = ("openai", "anthropic", "google")


@cache
def _client(provider: str):
    if provider == "openai":
        import openai

        return openai.OpenAI(max_retries=MAX_RETRIES)
    if provider == "anthropic":
        import anthropic

        return anthropic.Anthropic(max_retries=MAX_RETRIES)
    if provider == "google":
        from google import genai
        from google.genai import types

        retry = types.HttpRetryOptions(attempts=MAX_RETRIES, http_status_codes=[408, 429, 500, 502, 503, 504])
        return genai.Client(http_options=types.HttpOptions(retry_options=retry))
    raise ValueError(f"unknown provider {provider!r}; expected one of {PROVIDERS}")


def generate(model: dict, prompt: str, generation: dict, seed: int) -> dict:
    """Send `prompt` as the only user message (no system prompt) and return the answer.

    `model` is one entry of `models` in config.yaml; `generation` is the `generation` block.
    Returns `response_text`, `finish_reason` (normalised, see below), `provider_finish_reason` (as the API sent it),
    `n_output_tokens` and `model_version` (the exact version the API reports it served).

    finish_reason is one of:
      stop    the model finished its answer
      length  cut off at max_output_tokens
      blocked the provider's safety layer withheld or ended the answer (refusal stop reason, content filter, safety block)
    Any other reason is passed through unchanged, so nothing unexpected is hidden.
    """
    return {"openai": _openai, "anthropic": _anthropic, "google": _google}[model["provider"]](
        model, prompt, generation, seed
    )


def _openai(model: dict, prompt: str, generation: dict, seed: int) -> dict:
    response = _client("openai").chat.completions.create(
        model=model["id"],
        messages=[{"role": "user", "content": prompt}],
        temperature=generation["temperature"],
        max_completion_tokens=generation["max_output_tokens"],
        seed=seed,
    )
    choice = response.choices[0]
    raw = choice.finish_reason
    blocked = raw == "content_filter" or bool(choice.message.refusal)
    return {
        # A structured refusal arrives in `message.refusal` instead of `message.content`.
        "response_text": choice.message.content or choice.message.refusal or "",
        "finish_reason": "blocked" if blocked else {"stop": "stop", "length": "length"}.get(raw, raw),
        "provider_finish_reason": raw,
        "n_output_tokens": response.usage.completion_tokens,
        "model_version": response.model,
    }


def _anthropic(model: dict, prompt: str, generation: dict, seed: int) -> dict:
    response = _client("anthropic").messages.create(
        model=model["id"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=generation["max_output_tokens"],
        # SDK 1.x dropped the typed sampling arguments; Haiku 4.5 still accepts temperature in the request body.
        # The Messages API has no seed parameter.
        extra_body={"temperature": generation["temperature"]},
    )
    raw = response.stop_reason
    return {
        "response_text": "".join(block.text for block in response.content if block.type == "text"),
        "finish_reason": {"end_turn": "stop", "max_tokens": "length", "refusal": "blocked"}.get(raw, raw),
        "provider_finish_reason": raw,
        "n_output_tokens": response.usage.output_tokens,
        "model_version": response.model,
    }


# Gemini finish reasons that mean the safety layer stopped the answer.
_GOOGLE_BLOCKED = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"}


def _google(model: dict, prompt: str, generation: dict, seed: int) -> dict:
    from google.genai import types

    thinking = types.ThinkingConfig(thinking_level=model["thinking_level"]) if model.get("thinking_level") else None
    response = _client("google").models.generate_content(
        model=model["id"],
        contents=prompt,
        # Default safety settings on purpose: we test the model as shipped, filters included.
        config=types.GenerateContentConfig(
            temperature=generation["temperature"],
            max_output_tokens=generation["max_output_tokens"],
            seed=seed,
            thinking_config=thinking,
        ),
    )
    usage = response.usage_metadata
    candidate = response.candidates[0] if response.candidates else None
    if candidate is None:
        # The prompt itself was blocked before the model answered.
        block = response.prompt_feedback.block_reason if response.prompt_feedback else None
        raw = f"PROMPT_BLOCKED:{block.name if block else 'UNKNOWN'}"
        text, finish = "", "blocked"
    else:
        raw = candidate.finish_reason.name if candidate.finish_reason else None
        parts = candidate.content.parts if candidate.content and candidate.content.parts else []
        # Thought summaries are not part of the answer.
        text = "".join(part.text for part in parts if part.text and not part.thought)
        finish = "stop" if raw == "STOP" else "length" if raw == "MAX_TOKENS" else "blocked" if raw in _GOOGLE_BLOCKED else raw
    return {
        "response_text": text,
        "finish_reason": finish,
        "provider_finish_reason": raw,
        "n_output_tokens": (usage.candidates_token_count or 0) if usage else 0,
        "n_thinking_tokens": (usage.thoughts_token_count or 0) if usage else 0,
        "model_version": response.model_version,
    }


def count_tokens(model: dict, text: str) -> int:
    """Number of tokens `text` takes for this model, sent as a single user message.

    OpenAI is counted offline with tiktoken. Anthropic and Google have no public tokenizer, so their free
    token-counting endpoints are called (needs the API key). Anthropic's count includes a small fixed overhead for
    the message wrapper; it is the same for every prompt, so differences between tone versions are exact.
    """
    provider = model["provider"]
    if provider == "openai":
        import tiktoken

        return len(tiktoken.encoding_for_model(model["id"]).encode(text))
    if provider == "anthropic":
        return _client("anthropic").messages.count_tokens(
            model=model["id"], messages=[{"role": "user", "content": text}]
        ).input_tokens
    if provider == "google":
        return _client("google").models.count_tokens(model=model["id"], contents=text).total_tokens
    raise ValueError(f"unknown provider {provider!r}; expected one of {PROVIDERS}")
