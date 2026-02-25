from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from forge.data.datasets.sft_dataset import AlpacaToMessages

# TorchForge imports (names may vary slightly by version)
from forge.data.utils import mask_messages, TuneMessage


@dataclass
class AutoToMessages:
    """
    A "best-effort" message_transform similar in spirit to AlpacaToMessages, but
    it auto-detects common HF dataset schemas and converts them to:

        {"messages": List[TuneMessage]}

    Supported (auto-detected) styles:
      1) Chat messages list: {"messages": [{"role":..., "content":...}, ...]}
         (also supports common aliases: "content"/"text"/"value")
      2) ShareGPT: {"conversations": [{"from":"human","value":"..."}, {"from":"gpt","value":"..."}]}
      3) Alpaca: {"instruction":..., "input":..., "output":...}
      4) Prompt-completion: {"prompt":..., "completion":...} or {"prompt":..., "response":...}
      5) QA: {"question":..., "answer":...} or {"question":..., "answers": {... or [...]}}
         (+ optional "context"/"passage")
      6) Multiple-choice QA: {"question":..., "choices"/"options":..., "answerKey"/"label":...}
      7) Text-only fallback: {"text": "..."} (wraps as user->assistant with empty assistant; usually skipped)

    Field mapping (Option 1 - explicit override):
      Use `field_map` to pin exact field names when auto-detection is wrong or ambiguous.
      Supported keys:
        - "user"        : field to use as the user/instruction content (required)
        - "assistant"   : field to use as the assistant/output content (required)
        - "input"       : optional secondary user context appended to "user" field
        - "system"      : optional system prompt field

      Example:
        AutoToMessages(field_map={"user": "question", "assistant": "answer"})
        AutoToMessages(field_map={"user": "instruction", "input": "context", "assistant": "response"})

    Notes:
      - When `field_map` is set, auto-detection is completely bypassed.
      - Auto-detection can be wrong on ambiguous schemas. For best quality, use field_map.
      - This transform *does not tokenize*. It only builds TuneMessages and applies masking.
    """

    # How to mask for SFT: commonly "train_on_assistant"
    masking_strategy: str = "train_on_assistant"

    # If True, prepend a system message when present in sample
    # (e.g., sample["system"] or sample["system_prompt"])
    include_system: bool = True

    # If True, require an assistant message; otherwise returns None (skipped)
    require_assistant: bool = True

    # Explicit field mapping to bypass auto-detection.
    # Keys: "user" (required), "assistant" (required), "input" (optional), "system" (optional)
    # Example: {"user": "instruction", "assistant": "output", "input": "context"}
    field_map: Dict[str, str] = field(default_factory=dict)

    # Role normalization maps
    role_map_chat: Dict[str, str] = None
    role_map_sharegpt: Dict[str, str] = None

    def __post_init__(self) -> None:
        if self.field_map is None:
            self.field_map = {}
        if self.role_map_chat is None:
            self.role_map_chat = {
                "user": "user",
                "human": "user",
                "assistant": "assistant",
                "gpt": "assistant",
                "system": "system",
                "tool": "tool",
                "function": "tool",
            }
        if self.role_map_sharegpt is None:
            self.role_map_sharegpt = {
                "human": "user",
                "user": "user",
                "gpt": "assistant",
                "assistant": "assistant",
                "system": "system",
                "tool": "tool",
            }

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        msgs = self._convert(sample)
        if msgs is None:
            # TorchForge iterable dataset usually expects you to raise or skip.
            # Returning empty messages will likely error downstream, so raise.
            raise ValueError("AutoToMessages could not convert sample to messages.")

        # Apply masking in-place (train on assistant tokens typically)
        # Note: For Alpaca-style data, masking is already applied by AlpacaToMessages.
        # Calling mask_messages again with the same strategy is safe and idempotent.
        mask_messages(msgs, self.masking_strategy)
        return {"messages": msgs}

    # ----------------------------
    # Detection + Conversion
    # ----------------------------

    def _convert(self, sample: Dict[str, Any]) -> Optional[List[TuneMessage]]:
        # -1) Explicit field_map takes priority over all auto-detection
        if self.field_map:
            return self._from_field_map(sample)

        # 0) Optional system prompt fields
        sys_msg = None
        if self.include_system:
            sys_text = self._first_str(
                sample,
                ["system", "system_prompt", "systemPrompt", "instruction_system"],
            )
            if sys_text:
                sys_msg = TuneMessage(role="system", content=sys_text, eot=True)

        # 1) Chat messages list (already in messages)
        if isinstance(sample.get("messages"), list) and sample["messages"]:
            msgs = self._from_chat_messages(sample["messages"])
            if msgs:
                if sys_msg and (not msgs or msgs[0].role != "system"):
                    msgs.insert(0, sys_msg)
                return self._validate(msgs)

        # 2) ShareGPT-style "conversations"
        if isinstance(sample.get("conversations"), list) and sample["conversations"]:
            msgs = self._from_sharegpt(sample["conversations"])
            if msgs:
                if sys_msg and (not msgs or msgs[0].role != "system"):
                    msgs.insert(0, sys_msg)
                return self._validate(msgs)

        # 3) Alpaca-style
        if any(k in sample for k in ("instruction", "output")):
            inst = self._as_str(sample.get("instruction", ""))
            out = self._as_str(sample.get("output", ""))
            if inst and out:
                # Use the existing AlpacaToMessages implementation
                alpaca_transform = AlpacaToMessages(
                    column_map=None,  # Use default field names
                    masking_strategy=self.masking_strategy,
                )
                result = alpaca_transform(sample)
                msgs = result.get("messages", [])
                if msgs:
                    # Add system message if needed
                    if sys_msg and (not msgs or msgs[0].role != "system"):
                        msgs.insert(0, sys_msg)
                    # Note: AlpacaToMessages already applied masking, so messages are ready
                    return self._validate(msgs)

        # 4) Prompt-completion
        prompt = self._first_str(sample, ["prompt", "instruction_prompt", "query"])
        completion = self._first_str(
            sample, ["completion", "response", "answer", "output_text"]
        )
        if prompt and completion:
            msgs = [
                TuneMessage(role="user", content=prompt, eot=True),
                TuneMessage(role="assistant", content=completion, eot=True),
            ]
            if sys_msg:
                msgs.insert(0, sys_msg)
            return self._validate(msgs)

        # 5) QA
        question = self._first_str(sample, ["question", "query", "prompt_question"])
        if question:
            context = self._first_str(
                sample, ["context", "passage", "article", "paragraph"]
            )
            answer = self._extract_answer(sample)
            if answer:
                user = question if not context else f"{question}\n\nContext:\n{context}"
                msgs = [
                    TuneMessage(role="user", content=user, eot=True),
                    TuneMessage(role="assistant", content=answer, eot=True),
                ]
                if sys_msg:
                    msgs.insert(0, sys_msg)
                return self._validate(msgs)

        # 6) Multiple-choice QA
        if question and any(k in sample for k in ("choices", "options")):
            choices_txt = self._format_choices(
                sample.get("choices", None) or sample.get("options", None)
            )
            key = self._first_str(
                sample, ["answerKey", "label", "answer_key", "correct"]
            )
            if choices_txt and key:
                user = f"{question}\n\nChoices:\n{choices_txt}"
                msgs = [
                    TuneMessage(role="user", content=user, eot=True),
                    TuneMessage(role="assistant", content=key, eot=True),
                ]
                if sys_msg:
                    msgs.insert(0, sys_msg)
                return self._validate(msgs)

        # 7) Text-only fallback (usually not good for SFT)
        text = self._first_str(sample, ["text", "content", "raw_text"])
        if text:
            # If you really want to keep this, you could wrap it as user->assistant,
            # but this is usually pretraining data, not SFT.
            if self.require_assistant:
                return None
            msgs = [
                TuneMessage(
                    role="user",
                    content="Rewrite/summarize the following text:",
                    eot=True,
                ),
                TuneMessage(role="assistant", content=text, eot=True),
            ]
            if sys_msg:
                msgs.insert(0, sys_msg)
            return self._validate(msgs)

        return None

    # ----------------------------
    # Helpers
    # ----------------------------

    def _from_field_map(self, sample: Dict[str, Any]) -> Optional[List[TuneMessage]]:
        """Convert sample using explicit field_map, bypassing auto-detection.

        field_map keys:
          - "user"      (required): field name for the user/instruction content
          - "assistant" (required): field name for the assistant/output content
          - "input"     (optional): field name for secondary context appended to user
          - "system"    (optional): field name for system prompt
        """
        user_key = self.field_map.get("user")
        assistant_key = self.field_map.get("assistant")

        if not user_key or not assistant_key:
            raise ValueError(
                f"field_map must contain 'user' and 'assistant' keys. Got: {self.field_map}"
            )

        user_text = self._as_str(sample.get(user_key, ""))
        assistant_text = self._as_str(sample.get(assistant_key, ""))

        if not user_text or not assistant_text:
            return None

        # Append optional input/context to user message
        input_key = self.field_map.get("input")
        if input_key:
            input_text = self._as_str(sample.get(input_key, ""))
            if input_text:
                user_text = f"{user_text}\n\n{input_text}"

        msgs: List[TuneMessage] = []

        # Optional system message
        system_key = self.field_map.get("system")
        if system_key:
            system_text = self._as_str(sample.get(system_key, ""))
            if system_text:
                msgs.append(TuneMessage(role="system", content=system_text, eot=True))
        elif self.include_system:
            sys_text = self._first_str(
                sample, ["system", "system_prompt", "systemPrompt"]
            )
            if sys_text:
                msgs.append(TuneMessage(role="system", content=sys_text, eot=True))

        msgs.append(TuneMessage(role="user", content=user_text, eot=True))
        msgs.append(TuneMessage(role="assistant", content=assistant_text, eot=True))
        return self._validate(msgs)

    def _from_chat_messages(self, raw: List[Dict[str, Any]]) -> List[TuneMessage]:
        msgs: List[TuneMessage] = []
        for m in raw:
            if not isinstance(m, dict):
                continue
            role_raw = self._as_str(m.get("role", m.get("from", ""))).lower()
            role = self.role_map_chat.get(role_raw, role_raw or "user")

            # content field varies across datasets
            content = (
                m.get("content", None)
                if m.get("content", None) is not None
                else m.get("text", None)
                if m.get("text", None) is not None
                else m.get("value", None)
            )
            content_s = self._as_str(content)

            if not content_s:
                continue
            msgs.append(TuneMessage(role=role, content=content_s, eot=True))
        return msgs

    def _from_sharegpt(self, raw: List[Dict[str, Any]]) -> List[TuneMessage]:
        msgs: List[TuneMessage] = []
        for m in raw:
            if not isinstance(m, dict):
                continue
            src = self._as_str(m.get("from", "")).lower()
            role = self.role_map_sharegpt.get(src, "user")
            content_s = self._as_str(
                m.get("value", m.get("content", m.get("text", "")))
            )
            if not content_s:
                continue
            msgs.append(TuneMessage(role=role, content=content_s, eot=True))
        return msgs

    def _extract_answer(self, sample: Dict[str, Any]) -> str:
        # Common patterns: answer (str), answers (dict with "text"), answers (list), final_answer
        ans = sample.get("answer", None)
        if isinstance(ans, str) and ans.strip():
            return ans.strip()

        ans = sample.get("final_answer", None)
        if isinstance(ans, str) and ans.strip():
            return ans.strip()

        answers = sample.get("answers", None)
        if isinstance(answers, dict):
            txt = answers.get("text", None)
            if isinstance(txt, list) and txt:
                return self._as_str(txt[0]).strip()
            if isinstance(txt, str) and txt.strip():
                return txt.strip()
        if isinstance(answers, list) and answers:
            return self._as_str(answers[0]).strip()

        # Some datasets use "output" for answer even in QA
        out = sample.get("output", None)
        if isinstance(out, str) and out.strip():
            return out.strip()

        return ""

    def _format_choices(self, choices: Any) -> str:
        # Support: list of strings, list of dicts, dict of label->text
        if choices is None:
            return ""
        if isinstance(choices, dict):
            lines = [f"{k}: {self._as_str(v)}" for k, v in choices.items()]
            return "\n".join([ln for ln in lines if ln.strip()])
        if isinstance(choices, list):
            lines = []
            for i, c in enumerate(choices):
                if isinstance(c, str):
                    lines.append(f"{i}: {c}")
                elif isinstance(c, dict):
                    # common: {"label":"A","text":"..."} or {"key":"A","value":"..."}
                    label = self._as_str(c.get("label", c.get("key", str(i))))
                    text = self._as_str(c.get("text", c.get("value", "")))
                    if text:
                        lines.append(f"{label}: {text}")
            return "\n".join(lines)
        return ""

    def _validate(self, msgs: List[TuneMessage]) -> Optional[List[TuneMessage]]:
        if not msgs:
            return None
        if self.require_assistant and not any(m.role == "assistant" for m in msgs):
            return None
        # Basic sanity: no empty contents (content may be str or list)
        def _has_content(m: TuneMessage) -> bool:
            c = getattr(m, "content", "")
            if isinstance(c, str):
                return bool(c.strip())
            return bool(c)  # list/other: truthy if non-empty

        msgs = [m for m in msgs if _has_content(m)]

        # Enforce valid role set and alternation required by the chat template.
        # Step 1: remap non-standard roles.
        #   - "thinking" / "think" / "reasoning" → merge into the NEXT assistant
        #     message (prepend as <think>…</think>), then drop.
        #   - Any other unknown role → treat as assistant (best-effort).
        KNOWN_ROLES = {"system", "user", "assistant"}
        THINK_ROLES = {"thinking", "think", "reasoning", "thought"}
        normalized: List[TuneMessage] = []
        pending_think: List[str] = []
        for m in msgs:
            if m.role in THINK_ROLES:
                pending_think.append(self._as_str(getattr(m, "content", "")))
            elif m.role == "assistant":
                if pending_think:
                    think_block = "<think>" + "\n".join(pending_think) + "</think>\n"
                    pending_think = []
                    new_content = think_block + self._as_str(getattr(m, "content", ""))
                    m = TuneMessage(
                        role="assistant",
                        content=new_content,
                        eot=getattr(m, "eot", True),
                    )
                normalized.append(m)
            else:
                if pending_think:
                    # Orphan thinking block with no following assistant — discard
                    pending_think = []
                if m.role not in KNOWN_ROLES:
                    m = TuneMessage(
                        role="assistant",
                        content=getattr(m, "content", ""),
                        eot=getattr(m, "eot", True),
                    )
                normalized.append(m)
        msgs = [m for m in normalized if _has_content(m)]

        # Step 2: system message must be at position 0 only.
        # If system appears mid-conversation, drop or merge into user.
        fixed: List[TuneMessage] = []
        for i, m in enumerate(msgs):
            if m.role == "system" and i > 0:
                # merge into previous user message if possible, otherwise drop
                continue
            fixed.append(m)
        msgs = fixed

        # Step 3: collapse consecutive same-role messages by concatenating content.
        collapsed: List[TuneMessage] = []
        for m in msgs:
            if collapsed and collapsed[-1].role == m.role:
                prev = collapsed[-1]
                combined = (
                    self._as_str(getattr(prev, "content", ""))
                    + "\n"
                    + self._as_str(getattr(m, "content", ""))
                )
                collapsed[-1] = TuneMessage(
                    role=prev.role, content=combined, eot=getattr(m, "eot", True)
                )
            else:
                collapsed.append(m)
        msgs = collapsed

        # Step 4: conversation must start with user (after optional system).
        # Drop leading assistant messages.
        start = 0
        if msgs and msgs[0].role == "system":
            start = 1
        while start < len(msgs) and msgs[start].role != "user":
            start += 1
        msgs = (msgs[:1] if (msgs and msgs[0].role == "system") else []) + msgs[start:]

        if self.require_assistant and not any(m.role == "assistant" for m in msgs):
            return None
        return msgs if msgs else None

    def _first_str(self, d: Dict[str, Any], keys: List[str]) -> str:
        for k in keys:
            v = d.get(k, None)
            s = self._as_str(v)
            if s:
                return s
        return ""

    @staticmethod
    def _as_str(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, str):
            return x.strip()
        return str(x).strip()
