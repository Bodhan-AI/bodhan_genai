"""IndicBlockOCR: crops in, transcriptions out.

Heavy imports live inside methods, so importing this module stays free -- asserted by
tests/ocr/test_ocr_lazy_import.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Protocol, runtime_checkable

from bodhan_genai.ocr.engine.types import CropConfig, RecognizerConfig

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image


class CropRequest(NamedTuple):
    image: Image
    prompt: str


@runtime_checkable
class RecognizerBackend(Protocol):
    """``transcribe`` returns one string per request, in the same order."""

    def transcribe(self, requests: list[CropRequest]) -> list[str]: ...

    def close(self) -> None: ...


def build_requests(blocks, page, crop_cfg: CropConfig, table_format) -> tuple[list, list]:
    """Crop each block and pair it with its prompt.

    Returns ``(requests, orders)`` -- the reading-order rank of each request, so transcriptions
    can be matched back. Blocks that yield no crop are simply absent from both.
    """
    from bodhan_genai.ocr.engine.crops import area_clamp, crop_for
    from bodhan_genai.ocr.templates.contract import prompt_for

    requests, orders = [], []
    for block in blocks:
        crop = crop_for(block, page, crop_cfg)
        if crop is None:
            continue
        requests.append(
            CropRequest(area_clamp(crop, crop_cfg), prompt_for(block.type, table_format))
        )
        orders.append(block.order)
    return requests, orders


class HfRecognizer:
    """Reference recognizer on plain ``transformers`` -- no vLLM.

    Exists so IndicOCR can run anywhere ``transformers`` runs, including straight from the
    Hub with ``trust_remote_code=True``. It is the *quickstart* path, not the working one:
    without continuous batching it is orders of magnitude slower per block than
    :class:`VllmRecognizer`, so use it to try a page, not to parse a corpus.

    Output also diverges slightly from the vLLM path. Both decode greedily, but different kernels
    give different logits, and a near-tie flips the argmax -- so do not expect byte-identical
    transcriptions between the two backends.
    """

    def __init__(
        self,
        ckpt: str | None = None,
        config: RecognizerConfig | None = None,
        device: str = "auto",
        attn_implementation: str = "sdpa",
        batch_size: int = 8,
    ) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        from bodhan_genai.ocr.engine.checkpoints import resolve_ckpt

        self._torch = torch
        self.config = config or RecognizerConfig()
        # RecognizerConfig.batch_size sizes a vLLM chunk (~2048). Generating that many at once
        # here would simply OOM; HF batches are bounded by memory, not by scheduler behaviour.
        self.batch_size = batch_size
        self.ckpt = resolve_ckpt("recognizer", ckpt)

        self.processor = AutoProcessor.from_pretrained(self.ckpt)
        tokenizer = self.processor.tokenizer
        # Left padding so every sequence in a batch ends flush against the generation boundary.
        tokenizer.padding_side = "left"

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.ckpt,
            dtype=getattr(torch, self.config.dtype),
            device_map=device,
            attn_implementation=attn_implementation,
        )
        self.model.eval()

        # The checkpoint's generation_config carries eos_token_id 248044, which is an ordinary
        # word piece, not a turn terminator. Left alone, generate() never stops and every block
        # runs to max_new_tokens, repeating itself. vLLM does not hit this because it takes the
        # tokenizer's EOS. Trust the tokenizer here too.
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def _prompt(self, text: str) -> str:
        return self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}],
            add_generation_prompt=True,
            tokenize=False,
        )

    def transcribe(self, requests: list[CropRequest]) -> list[str]:
        texts: list[str] = []
        for i in range(0, len(requests), self.batch_size):
            chunk = requests[i : i + self.batch_size]
            inputs = self.processor(
                text=[self._prompt(r.prompt) for r in chunk],
                images=[r.image for r in chunk],
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)
            prompt_len = inputs["input_ids"].shape[-1]

            with self._torch.inference_mode():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    do_sample=False,
                    use_cache=True,
                    eos_token_id=self.eos_token_id,
                    pad_token_id=self.pad_token_id,
                )
            texts.extend(self.processor.batch_decode(out[:, prompt_len:], skip_special_tokens=True))
        return [t.strip() for t in texts]

    def close(self) -> None:
        self.model = None
        self.processor = None
