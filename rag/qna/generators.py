from __future__ import annotations

import requests
import logging
import threading
import time
from typing import Any

log = logging.getLogger(__name__)

_thread_local = threading.local()

RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}

def retry_delay(attempt: int, response: requests.Response | None = None) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")

        if retry_after:
            try:
                return min(float(retry_after), 60.0)
            except ValueError:
                pass
    return min(2.0**attempt, 30.0)

def get_http_session() -> requests.Session:
    session = getattr(_thread_local, "http_session", None)

    if session is None:
        session = requests.Session()
        _thread_local.http_session = session
    return session

def ollama_generate(base_url: str, model: str, prompt: str, *, retries: int = 3, keep_alive: str = "10m", timeout: int = 300, connect_timeout: int = 15, options: dict[str, Any] | None = None, think: bool | str | None = None) -> str:
    if retries < 1:
        raise ValueError(f"retries must be at least 1, got {retries}")

    session = get_http_session()
    url = f"{base_url.rstrip('/')}/api/generate"

    request_options: dict[str, Any] = {
        "temperature": 0.0,
        "top_p": 0.9,
    }

    if options:
        request_options.update(options)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": keep_alive,
        "options": request_options,
    }
    if think is not None:
        payload["think"] = think

    last_error: Exception | None = None

    for attempt in range(retries):
        started = time.perf_counter()
        response: requests.Response | None = None

        try:
            log.info("[OLLAMA] request model=%s " "attempt=%d/%d prompt_chars=%d " "num_ctx=%s num_predict=%s", model, attempt + 1, retries, len(prompt), request_options.get("num_ctx"), request_options.get("num_predict"))
            response = session.post(url, json=payload, timeout=(connect_timeout, timeout))
            if (response.status_code in RETRYABLE_STATUS_CODES):
                raise requests.HTTPError(f"Retryable HTTP status " f"{response.status_code}", response=response,)

            response.raise_for_status()
            data = response.json()
            thinking_trace = str(data.get("thinking") or "").strip()
            raw_answer = str(data.get("response") or "").strip()
            done_reason = str(data.get("done_reason") or "").strip()
            answer = strip_thinking_blocks(str(data.get("response", "")))

            if not answer:
                raise RuntimeError("[OLLAMA] Ollama returned an empty response")

            elapsed = time.perf_counter() - started
            if (think is not None) and (think == True or "true" in think): 
                log.info("[OLLAMA_THINK] model=%s think=%r thinking_chars=%d response_chars=%d raw_response_chars=%d prompt_tokens=%s output_tokens=%s total_duration=%.2fs", model, think, len(thinking_trace), len(answer), len(raw_answer), data.get("prompt_eval_count"), data.get("eval_count"), float(data.get("total_duration", 0)) / 1_000_000_000)
            else:
                log.info("[OLLAMA] completed model=%s elapsed=%.2fs prompt_tokens=%s output_tokens=%s total_duration=%.2fs", model, elapsed, data.get("prompt_eval_count"), data.get("eval_count"), float(data.get("total_duration", 0)) / 1_000_000_000)
            
            if not answer:
                if thinking_trace or "<think>" in raw_answer.lower() or done_reason == "length":
                    raise ValueError(f"Ollama produced thinking but no final answer: model={model!r}, think={think!r}, done_reason={done_reason!r}, thinking_chars={len(thinking_trace)}, raw_response_chars={len(raw_answer)}")
                raise ValueError("Ollama returned an empty response")
            
            return answer

        except (requests.ConnectTimeout, requests.ReadTimeout, requests.ConnectionError, requests.HTTPError, ValueError, RuntimeError,) as exc:
            last_error = exc
            retryable = isinstance(exc, (requests.ConnectTimeout, requests.ReadTimeout, requests.ConnectionError,))

            if isinstance(exc, requests.HTTPError):
                status = (exc.response.status_code if exc.response is not None else None)
                retryable = (status in RETRYABLE_STATUS_CODES)

            if (not retryable or attempt + 1 >= retries):
                break

            delay = retry_delay(attempt, response,)
            log.warning("[OLLAMA] request failed " "model=%s attempt=%d/%d " "error=%s: %s retrying_in=%.1fs", model, attempt + 1, retries, type(exc).__name__, exc, delay)
            time.sleep(delay)

    raise RuntimeError("Ollama generation failed after " f"{retries} attempts: " f"model={model!r}, " f"prompt_chars={len(prompt)}, " f"last_error={last_error}") from last_error

def llamacpp_generate(base_url: str, model: str, prompt: str, *, timeout: int, connect_timeout: int = 15, retries: int = 8, temperature: float = 0.0, top_p: float = 0.9, max_tokens: int | None = None) -> str:
    if retries < 1:
        raise ValueError(f"retries must be at least 1, got {retries}")

    session = get_http_session()
    url = (
        f"{base_url.rstrip('/')}"
        "/v1/chat/completions"
    )

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }

    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)

    last_error: Exception | None = None

    for attempt in range(retries):
        started = time.perf_counter()
        response: requests.Response | None = None

        try:
            log.info(
                "[LLAMACPP] request model=%s "
                "attempt=%d/%d prompt_chars=%d",
                model,
                attempt + 1,
                retries,
                len(prompt),
            )

            response = session.post(
                url,
                json=payload,
                timeout=(
                    connect_timeout,
                    timeout,
                ),
            )

            if (
                response.status_code
                in RETRYABLE_STATUS_CODES
            ):
                raise requests.HTTPError(
                    f"Retryable HTTP status "
                    f"{response.status_code}",
                    response=response,
                )

            response.raise_for_status()

            data = response.json()

            choices = data.get("choices") or []

            if not choices:
                raise RuntimeError(
                    "llama.cpp returned no choices"
                )

            answer = str(
                choices[0]
                .get("message", {})
                .get("content", "")
            ).strip()

            if not answer:
                raise RuntimeError(
                    "llama.cpp returned an empty response"
                )

            log.info(
                "[LLAMACPP] completed model=%s "
                "elapsed=%.2fs usage=%s",
                model,
                time.perf_counter() - started,
                data.get("usage"),
            )

            return answer

        except (
            requests.ConnectTimeout,
            requests.ReadTimeout,
            requests.ConnectionError,
            requests.HTTPError,
            ValueError,
            KeyError,
            IndexError,
            RuntimeError,
        ) as exc:
            last_error = exc

            retryable = isinstance(
                exc,
                (
                    requests.ConnectTimeout,
                    requests.ReadTimeout,
                    requests.ConnectionError,
                ),
            )

            if isinstance(exc, requests.HTTPError):
                status = (
                    exc.response.status_code
                    if exc.response is not None
                    else None
                )

                retryable = (
                    status in RETRYABLE_STATUS_CODES
                )

            if (
                not retryable
                or attempt + 1 >= retries
            ):
                break

            delay = retry_delay(
                attempt,
                response,
            )

            log.warning(
                "[LLAMACPP] request failed "
                "model=%s attempt=%d/%d "
                "error=%s: %s retrying_in=%.1fs",
                model,
                attempt + 1,
                retries,
                type(exc).__name__,
                exc,
                delay,
            )

            time.sleep(delay)

    raise RuntimeError(
        "llama.cpp generation failed after "
        f"{retries} attempts: "
        f"model={model!r}, "
        f"prompt_chars={len(prompt)}, "
        f"last_error={last_error}"
    ) from last_error

def strip_thinking_blocks(text: str) -> str:
    import re
    text = text or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()

def generate(cfg: dict, prompt: str, *, retries: int | None = None, timeout: int | None = None, model_override: str | None = None, provider_override: str | None = None, options_override: dict[str, Any] | None = None, think_override: bool | str | None = None) -> str:
    runtime = cfg.get("runtime", {}) or {}

    provider = str(provider_override or runtime.get("qa_provider", "ollama")).strip().lower()

    if provider in {"llama.cpp", "llama-cpp"}:
        provider = "llamacpp"

    if provider == "ollama":
        ollama = cfg.get("ollama", {}) or {}
        model = str(model_override or ollama.get("qa_model") or ollama.get("model") or "llama3.2:3b")
        effective_retries = int(retries if retries is not None else ollama.get("qa_retries", 3))
        effective_timeout = int(timeout if timeout is not None else ollama.get("qa_timeout", ollama.get("timeout", 300)))
        connect_timeout = int(ollama.get("connect_timeout", 15))
        think = think_override if think_override is not None else ollama.get("qa_think", None)

        options: dict[str, Any] = {
            "temperature": float(ollama.get("qa_temperature", 0.0)),
            "top_p": float(ollama.get("qa_top_p", 0.9,)),
            "num_ctx": int(ollama.get("qa_num_ctx", 16384)),
            "num_predict": int(ollama.get("qa_num_predict", 1024)),
        }

        optional_option_names = (
            "seed",
            "repeat_penalty",
            "top_k",
            "min_p",
        )

        for option_name in optional_option_names:
            config_name = f"qa_{option_name}"
            if config_name in ollama:
                options[option_name] = ollama[config_name]

        if options_override:
            options.update(options_override)

        return ollama_generate(
            str(ollama.get("base_url", "http://localhost:11434")).strip(),
            model,
            prompt,
            retries=effective_retries,
            keep_alive=str(ollama.get("qa_keep_alive", "10m")),
            timeout=effective_timeout,
            connect_timeout=connect_timeout,
            options=options,
            think=think,
        )

    if provider == "llamacpp":
        lc = cfg.get("llamacpp", {}) or {}

        model = str(
            model_override
            or lc.get("qa_model")
            or lc.get("model")
            or ""
        )

        if not model:
            raise RuntimeError(
                "No llama.cpp QA model configured"
            )

        effective_retries = int(
            retries
            if retries is not None
            else lc.get("qa_retries", 3)
        )

        effective_timeout = int(
            timeout
            if timeout is not None
            else lc.get("timeout", 300)
        )

        return llamacpp_generate(
            str(lc["base_url"]),
            model,
            prompt,
            retries=effective_retries,
            timeout=effective_timeout,
            connect_timeout=int(
                lc.get("connect_timeout", 15)
            ),
            temperature=float(
                lc.get("qa_temperature", 0.0)
            ),
            top_p=float(
                lc.get("qa_top_p", 0.9)
            ),
            max_tokens=(
                int(lc["qa_max_tokens"])
                if lc.get("qa_max_tokens")
                is not None
                else None
            ),
        )

    raise RuntimeError(
        f"Unknown QA provider: {provider}"
    )