"""FastAPI server — OpenAI-compatible API for DeepSeek models."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .config import InferenceConfig
from .inference import DeepSeekInferenceEngine

logger = logging.getLogger(__name__)
app = FastAPI(title="DeepSeek ROCm API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: Optional[DeepSeekInferenceEngine] = None


# ── Request / Response Models ───────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "deepseek-v3"
    messages: list[Message]
    max_tokens: int = Field(default=512, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = False
    stop: Optional[list[str]] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatChoice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: Usage


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class StreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": engine is not None}


@app.get("/v1/models")
async def list_models():
    return {
        "data": [
            {"id": "deepseek-v3", "object": "model", "owned_by": "deepseek"},
            {"id": "deepseek-r1", "object": "model", "owned_by": "deepseek"},
            {"id": "deepseek-v2", "object": "model", "owned_by": "deepseek"},
        ]
    }


def _build_prompt(messages: list[Message]) -> str:
    parts = []
    for m in messages:
        parts.append(f"<|{m.role}|>\n{m.content}")
    parts.append("<|assistant|>")
    return "\n".join(parts)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest):
    if engine is None:
        raise HTTPException(503, "Model not loaded")

    prompt = _build_prompt(req.messages)

    if req.stream:
        return StreamingResponse(
            _stream_response(prompt, req),
            media_type="text/event-stream",
        )

    text, stats = engine.generate(
        prompt,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        stop_sequences=req.stop,
    )

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=req.model,
        choices=[ChatChoice(message=Message(role="assistant", content=text))],
        usage=Usage(
            prompt_tokens=stats.prompt_tokens,
            completion_tokens=stats.generated_tokens,
            total_tokens=stats.prompt_tokens + stats.generated_tokens,
        ),
    )


async def _stream_response(prompt: str, req: ChatCompletionRequest):
    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    for token in engine.stream_generate(prompt, req.max_tokens, req.temperature, req.top_p):
        chunk = StreamResponse(
            id=req_id,
            created=created,
            model=req.model,
            choices=[StreamChoice(delta=DeltaMessage(content=token))],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"
        await asyncio.sleep(0)

    final = StreamResponse(
        id=req_id,
        created=created,
        model=req.model,
        choices=[StreamChoice(delta=DeltaMessage(), finish_reason="stop")],
    )
    yield f"data: {final.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


# ── Entrypoint ──────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    global engine
    config = InferenceConfig.from_yaml(args.config)
    engine = DeepSeekInferenceEngine(config)

    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)


if __name__ == "__main__":
    main()
