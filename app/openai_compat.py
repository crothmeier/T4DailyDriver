"""
Comprehensive OpenAI API compatibility layer for vLLM service.
Provides full /v1/chat/completions and /v1/completions endpoints with SSE streaming,
token counting, usage statistics, model name aliasing, and function calling support.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken not available - falling back to simple token counting")

logger = logging.getLogger(__name__)


# OpenAI API Models
class ChatMessage(BaseModel):
    """Chat message in OpenAI format."""

    role: str = Field(..., description="Role: system, user, assistant, or function")
    content: str | None = Field(None, description="Message content")
    name: str | None = Field(None, description="Name of the function or user")
    function_call: dict[str, Any] | None = Field(None, description="Function call data")


class Function(BaseModel):
    """Function definition for function calling."""

    name: str = Field(..., description="Function name")
    description: str | None = Field(None, description="Function description")
    parameters: dict[str, Any] = Field(..., description="JSON schema for function parameters")


class ChatCompletionRequest(BaseModel):
    """OpenAI chat completion request."""

    model: str = Field(..., description="Model name")
    messages: list[ChatMessage] = Field(..., description="List of messages")
    functions: list[Function] | None = Field(None, description="Available functions")
    function_call: str | dict[str, str] | None = Field(None, description="Function call control")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    n: int = Field(1, ge=1, le=10)
    stream: bool = Field(False)
    stop: str | list[str] | None = Field(None)
    max_tokens: int | None = Field(None, ge=1)
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    logit_bias: dict[str, float] | None = Field(None)
    user: str | None = Field(None)


class CompletionRequest(BaseModel):
    """OpenAI text completion request."""

    model: str = Field(..., description="Model name")
    prompt: str | list[str] | list[int] | list[list[int]] = Field(..., description="Prompt(s)")
    suffix: str | None = Field(None)
    max_tokens: int = Field(16, ge=1)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    n: int = Field(1, ge=1, le=10)
    stream: bool = Field(False)
    logprobs: int | None = Field(None, ge=0, le=5)
    echo: bool = Field(False)
    stop: str | list[str] | None = Field(None)
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    best_of: int = Field(1, ge=1)
    logit_bias: dict[str, float] | None = Field(None)
    user: str | None = Field(None)


class Usage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    """Chat completion choice."""

    index: int
    message: ChatMessage
    finish_reason: str | None = None


class CompletionChoice(BaseModel):
    """Text completion choice."""

    text: str
    index: int
    logprobs: dict[str, Any] | None = None
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    """OpenAI chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


class CompletionResponse(BaseModel):
    """OpenAI text completion response."""

    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Usage


class StreamingChoice(BaseModel):
    """Streaming response choice."""

    index: int
    delta: dict[str, Any]
    finish_reason: str | None = None


class StreamingResponse(BaseModel):
    """Streaming response format."""

    id: str
    object: str
    created: int
    model: str
    choices: list[StreamingChoice]


@dataclass
class TokenCounter:
    """Token counting utility with tiktoken integration."""

    def __init__(self):
        self.encoder = None
        if TIKTOKEN_AVAILABLE:
            try:
                # Use cl100k_base encoding (GPT-4/GPT-3.5-turbo)
                self.encoder = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                logger.warning(f"Failed to initialize tiktoken encoder: {e}")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.encoder:
            try:
                return len(self.encoder.encode(text))
            except Exception as e:
                logger.warning(f"tiktoken encoding failed: {e}")

        # Fallback to simple word counting
        return max(1, len(text.split()) + len(text) // 4)

    def count_message_tokens(self, messages: list[ChatMessage]) -> int:
        """Count tokens in chat messages."""
        total = 0

        for message in messages:
            # Base tokens per message (role, formatting)
            total += 4

            if message.content:
                total += self.count_tokens(message.content)
            if message.name:
                total += self.count_tokens(message.name)
            if message.function_call:
                total += self.count_tokens(json.dumps(message.function_call))

        # Add tokens for conversation structure
        total += 2

        return total


class ModelNameMapper:
    """Maps OpenAI model names to internal model configurations."""

    def __init__(self, model_path: str):
        self.internal_model = model_path
        self.aliases = {
            "gpt-3.5-turbo": model_path,
            "gpt-3.5-turbo-0613": model_path,
            "gpt-3.5-turbo-16k": model_path,
            "gpt-4": model_path,
            "gpt-4-0613": model_path,
            "gpt-4-32k": model_path,
            "text-davinci-003": model_path,
            "text-davinci-002": model_path,
            "code-davinci-002": model_path,
            "text-curie-001": model_path,
            "text-babbage-001": model_path,
            "text-ada-001": model_path,
        }

    def resolve_model(self, requested_model: str) -> str:
        """Resolve requested model to internal model path."""
        return self.aliases.get(requested_model, self.internal_model)

    def get_response_model_name(self, requested_model: str) -> str:
        """Get the model name to return in responses."""
        # Return the requested name to maintain API compatibility
        return requested_model if requested_model in self.aliases else self.internal_model


class FunctionCallHandler:
    """Handles function calling for compatible models."""

    def __init__(self):
        self.supported_models = {"gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo-0613", "gpt-4-0613"}

    def is_function_calling_supported(self, model: str) -> bool:
        """Check if model supports function calling."""
        return model in self.supported_models

    def format_functions_in_prompt(self, messages: list[ChatMessage], functions: list[Function]) -> str:
        """Format functions and messages into a single prompt for non-function-calling models."""
        # Create function documentation
        function_docs = []
        for func in functions:
            doc = f"Function: {func.name}\n"
            if func.description:
                doc += f"Description: {func.description}\n"
            doc += f"Parameters: {json.dumps(func.parameters, indent=2)}\n"
            function_docs.append(doc)

        # Build prompt
        prompt_parts = []

        if function_docs:
            prompt_parts.append("Available functions:")
            prompt_parts.extend(function_docs)
            prompt_parts.append("\\nConversation:")

        # Add messages
        for msg in messages:
            role_map = {"system": "System", "user": "Human", "assistant": "Assistant"}
            role = role_map.get(msg.role, msg.role.title())

            if msg.content:
                prompt_parts.append(f"{role}: {msg.content}")
            elif msg.function_call:
                # Format function call
                func_call = json.dumps(msg.function_call)
                prompt_parts.append(f"{role}: [Function Call: {func_call}]")

        prompt_parts.append("Assistant:")
        return "\\n".join(prompt_parts)

    def parse_function_call_from_response(self, response_text: str) -> dict[str, Any] | None:
        """Parse function call from model response."""
        # Look for function call patterns
        patterns = [r"\\[Function Call: ({.*?})\\]", r"function_call\\s*:\\s*({.*?})", r"call_function\\(([^)]+)\\)"]

        import re

        for pattern in patterns:
            matches = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
            if matches:
                try:
                    return json.loads(matches.group(1))
                except json.JSONDecodeError:
                    continue

        return None


class OpenAICompatibilityLayer:
    """Main OpenAI compatibility layer."""

    def __init__(self, model_path: str, vllm_engine):
        self.model_path = model_path
        self.engine = vllm_engine
        self.token_counter = TokenCounter()
        self.model_mapper = ModelNameMapper(model_path)
        self.function_handler = FunctionCallHandler()

    async def chat_completion(
        self, request: ChatCompletionRequest, correlation_id: str | None = None
    ) -> ChatCompletionResponse | StreamingResponse:
        """Handle chat completion request."""

        # Resolve model
        resolved_model = self.model_mapper.resolve_model(request.model)
        response_model_name = self.model_mapper.get_response_model_name(request.model)

        # Count prompt tokens
        prompt_tokens = self.token_counter.count_message_tokens(request.messages)

        # Handle function calling
        if request.functions and self.function_handler.is_function_calling_supported(request.model):
            # Native function calling support
            prompt = self._format_chat_messages(request.messages)
        else:
            # Format functions into prompt for non-function-calling models
            if request.functions:
                prompt = self.function_handler.format_functions_in_prompt(request.messages, request.functions)
            else:
                prompt = self._format_chat_messages(request.messages)

        if request.stream:
            return await self._stream_chat_completion(
                request, prompt, prompt_tokens, response_model_name, correlation_id
            )
        else:
            return await self._batch_chat_completion(
                request, prompt, prompt_tokens, response_model_name, correlation_id
            )

    async def text_completion(
        self, request: CompletionRequest, correlation_id: str | None = None
    ) -> CompletionResponse | StreamingResponse:
        """Handle text completion request."""

        # Resolve model
        resolved_model = self.model_mapper.resolve_model(request.model)
        response_model_name = self.model_mapper.get_response_model_name(request.model)

        # Handle different prompt formats
        if isinstance(request.prompt, str):
            prompt = request.prompt
        elif isinstance(request.prompt, list):
            if request.prompt and isinstance(request.prompt[0], str):
                prompt = request.prompt[0]  # Take first prompt for now
            else:
                raise HTTPException(status_code=400, detail="Token array prompts not supported")
        else:
            raise HTTPException(status_code=400, detail="Invalid prompt format")

        prompt_tokens = self.token_counter.count_tokens(prompt)

        if request.stream:
            return await self._stream_text_completion(
                request, prompt, prompt_tokens, response_model_name, correlation_id
            )
        else:
            return await self._batch_text_completion(
                request, prompt, prompt_tokens, response_model_name, correlation_id
            )

    def _format_chat_messages(self, messages: list[ChatMessage]) -> str:
        """Format chat messages into a prompt."""
        prompt_parts = []

        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"Human: {msg.content}")
            elif msg.role == "assistant":
                if msg.content:
                    prompt_parts.append(f"Assistant: {msg.content}")
                elif msg.function_call:
                    func_call = json.dumps(msg.function_call)
                    prompt_parts.append(f"Assistant: [Function Call: {func_call}]")

        prompt_parts.append("Assistant:")
        return "\\n".join(prompt_parts)

    async def _batch_chat_completion(
        self,
        request: ChatCompletionRequest,
        prompt: str,
        prompt_tokens: int,
        model_name: str,
        correlation_id: str | None,
    ) -> ChatCompletionResponse:
        """Handle batch chat completion."""
        from vllm import SamplingParams

        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens or 256,
            stop=request.stop,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
        )

        # Generate completion
        request_id = correlation_id or str(uuid.uuid4())
        engine = await self.engine.get_connection()

        try:
            results_generator = engine.generate(prompt, sampling_params, request_id)

            final_output = None
            async for request_output in results_generator:
                final_output = request_output

            if not final_output or not final_output.outputs:
                raise HTTPException(status_code=500, detail="No output generated")

            output = final_output.outputs[0]
            response_text = output.text or ""
            completion_tokens = len(output.token_ids) if output.token_ids else 0

            # Check for function calls
            function_call = None
            if request.functions:
                function_call = self.function_handler.parse_function_call_from_response(response_text)

            # Create message
            message = ChatMessage(
                role="assistant", content=response_text if not function_call else None, function_call=function_call
            )

            # Create choice
            choice = ChatCompletionChoice(index=0, message=message, finish_reason="stop")

            # Create usage
            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )

            return ChatCompletionResponse(
                id=f"chatcmpl-{request_id[:8]}",
                created=int(time.time()),
                model=model_name,
                choices=[choice],
                usage=usage,
            )

        finally:
            await self.engine.release_connection()

    async def _stream_chat_completion(
        self,
        request: ChatCompletionRequest,
        prompt: str,
        prompt_tokens: int,
        model_name: str,
        correlation_id: str | None,
    ) -> StreamingResponse:
        """Handle streaming chat completion."""

        async def stream_generator():
            from vllm import SamplingParams

            request_id = correlation_id or str(uuid.uuid4())

            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens or 256,
                stop=request.stop,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
            )

            engine = await self.engine.get_connection()

            try:
                results_generator = engine.generate(prompt, sampling_params, request_id)

                last_text_length = 0
                completion_tokens = 0

                async for request_output in results_generator:
                    if request_output.outputs:
                        output = request_output.outputs[0]

                        if output.text:
                            # Stream incremental text
                            new_text = output.text[last_text_length:]
                            if new_text:
                                last_text_length = len(output.text)
                                completion_tokens = len(output.token_ids) if output.token_ids else 0

                                # Create streaming response
                                chunk = StreamingResponse(
                                    id=f"chatcmpl-{request_id[:8]}",
                                    object="chat.completion.chunk",
                                    created=int(time.time()),
                                    model=model_name,
                                    choices=[StreamingChoice(index=0, delta={"content": new_text}, finish_reason=None)],
                                )

                                yield f"data: {chunk.model_dump_json()}\\n\\n"

                        # Send keep-alive every 30 seconds
                        await asyncio.sleep(0.1)  # Small delay to prevent overwhelming

                # Send final chunk with usage
                final_chunk = StreamingResponse(
                    id=f"chatcmpl-{request_id[:8]}",
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model=model_name,
                    choices=[StreamingChoice(index=0, delta={}, finish_reason="stop")],
                )

                yield f"data: {final_chunk.model_dump_json()}\\n\\n"
                yield "data: [DONE]\\n\\n"

            except Exception as e:
                error_chunk = {"error": {"message": str(e), "type": "internal_error", "code": "internal_error"}}
                yield f"data: {json.dumps(error_chunk)}\\n\\n"
            finally:
                await self.engine.release_connection()

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-ID": correlation_id or str(uuid.uuid4()),
            },
        )

    async def _batch_text_completion(
        self,
        request: CompletionRequest,
        prompt: str,
        prompt_tokens: int,
        model_name: str,
        correlation_id: str | None,
    ) -> CompletionResponse:
        """Handle batch text completion."""
        from vllm import SamplingParams

        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
        )

        # Generate completion
        request_id = correlation_id or str(uuid.uuid4())
        engine = await self.engine.get_connection()

        try:
            results_generator = engine.generate(prompt, sampling_params, request_id)

            final_output = None
            async for request_output in results_generator:
                final_output = request_output

            if not final_output or not final_output.outputs:
                raise HTTPException(status_code=500, detail="No output generated")

            output = final_output.outputs[0]
            response_text = output.text or ""
            completion_tokens = len(output.token_ids) if output.token_ids else 0

            # Create choice
            choice = CompletionChoice(text=response_text, index=0, finish_reason="stop")

            # Create usage
            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )

            return CompletionResponse(
                id=f"cmpl-{request_id[:8]}", created=int(time.time()), model=model_name, choices=[choice], usage=usage
            )

        finally:
            await self.engine.release_connection()

    async def _stream_text_completion(
        self,
        request: CompletionRequest,
        prompt: str,
        prompt_tokens: int,
        model_name: str,
        correlation_id: str | None,
    ) -> StreamingResponse:
        """Handle streaming text completion."""

        async def stream_generator():
            from vllm import SamplingParams

            request_id = correlation_id or str(uuid.uuid4())

            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                stop=request.stop,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
            )

            engine = await self.engine.get_connection()

            try:
                results_generator = engine.generate(prompt, sampling_params, request_id)

                last_text_length = 0
                completion_tokens = 0

                async for request_output in results_generator:
                    if request_output.outputs:
                        output = request_output.outputs[0]

                        if output.text:
                            # Stream incremental text
                            new_text = output.text[last_text_length:]
                            if new_text:
                                last_text_length = len(output.text)
                                completion_tokens = len(output.token_ids) if output.token_ids else 0

                                # Create streaming response
                                chunk = {
                                    "id": f"cmpl-{request_id[:8]}",
                                    "object": "text_completion",
                                    "created": int(time.time()),
                                    "model": model_name,
                                    "choices": [{"text": new_text, "index": 0, "finish_reason": None}],
                                }

                                yield f"data: {json.dumps(chunk)}\\n\\n"

                # Send final chunk
                yield "data: [DONE]\\n\\n"

            except Exception as e:
                error_chunk = {"error": {"message": str(e), "type": "internal_error", "code": "internal_error"}}
                yield f"data: {json.dumps(error_chunk)}\\n\\n"
            finally:
                await self.engine.release_connection()

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-ID": correlation_id or str(uuid.uuid4()),
            },
        )
