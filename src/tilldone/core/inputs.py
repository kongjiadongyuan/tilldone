from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextInput:
    text: str
    kind: str = "text"


@dataclass(frozen=True)
class ImageInput:
    path: str
    kind: str = "image"


InputItem = TextInput | ImageInput


def normalize_prompt(prompt: "str | list[InputItem]") -> list[InputItem]:
    if isinstance(prompt, str):
        return [TextInput(text=prompt)]
    return list(prompt)
