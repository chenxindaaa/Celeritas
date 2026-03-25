from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import torch


THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent.parent
SERIALIZATION_DIR = ROOT_DIR / "test" / "testSerialization"
DEFAULT_CHECKPOINT = ROOT_DIR / "models" / "stories15M.pt"
DEFAULT_TOKENIZER = ROOT_DIR / "models" / "tokenizer.model"

if str(SERIALIZATION_DIR) not in sys.path:
    sys.path.insert(0, str(SERIALIZATION_DIR))

from load import load_checkpoint  # noqa: E402


def try_load_sentencepiece(tokenizer_path: Path):
    try:
        import sentencepiece as spm
    except ImportError:
        return None

    if not tokenizer_path.exists():
        return None

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(str(tokenizer_path))
    return tokenizer


def parse_token_ids(token_ids_text: str) -> List[int]:
    parts = [part.strip() for part in token_ids_text.split(",") if part.strip()]
    if not parts:
        raise ValueError("token id list is empty")
    return [int(part) for part in parts]


def encode_prompt(prompt: str,
                  tokenizer_path: Path,
                  explicit_token_ids: Optional[str]) -> List[int]:
    if explicit_token_ids:
        return parse_token_ids(explicit_token_ids)

    tokenizer = try_load_sentencepiece(tokenizer_path)
    if tokenizer is None:
        raise RuntimeError(
            "sentencepiece is unavailable or tokenizer.model is missing. "
            "Install sentencepiece or pass --token-ids."
        )

    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        raise ValueError("prompt encoded to an empty token sequence")
    return token_ids


def decode_tokens(token_ids: List[int], tokenizer_path: Path) -> str:
    tokenizer = try_load_sentencepiece(tokenizer_path)
    if tokenizer is None:
        return " ".join(str(token_id) for token_id in token_ids)
    return tokenizer.decode(token_ids)


def build_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load stories15M.pt and run inference.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="Path to the .pt checkpoint")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER,
                        help="Path to tokenizer.model")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                        help="Prompt text to generate from")
    parser.add_argument("--token-ids", type=str, default=None,
                        help="Comma-separated token ids, used when tokenizer is unavailable")
    parser.add_argument("--max-new-tokens", type=int, default=64,
                        help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature; use 0 for greedy decoding")
    parser.add_argument("--top-k", type=int, default=200,
                        help="Top-k sampling cutoff")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device, e.g. cpu or cuda")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    device = build_device(args.device)
    model = load_checkpoint(str(args.checkpoint))
    model = model.to(device)
    model.eval()

    prompt_token_ids = encode_prompt(args.prompt, args.tokenizer, args.token_ids)
    input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)

    with torch.inference_mode():
        generated = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    output_token_ids = generated[0].tolist()
    text = decode_tokens(output_token_ids, args.tokenizer)

    print(f"checkpoint: {args.checkpoint}")
    print(f"device: {device}")
    print(f"input token count: {len(prompt_token_ids)}")
    print(f"output token count: {len(output_token_ids)}")
    print("generated text:")
    print(text)


if __name__ == "__main__":
    main()
