from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent.parent
SERIALIZATION_DIR = ROOT_DIR / "test" / "testSerialization"
DEFAULT_CHECKPOINT = ROOT_DIR / "models" / "stories15M.pt"
DEFAULT_TOKENIZER = ROOT_DIR / "models" / "tokenizer.model"
DEFAULT_TOKENIZER_GOLDEN = THIS_DIR / "sentencepiece_golden.txt"
DEFAULT_EMBEDDING_GOLDEN = THIS_DIR / "embedding_golden.txt"

if str(SERIALIZATION_DIR) not in sys.path:
    sys.path.insert(0, str(SERIALIZATION_DIR))


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


def encode_prompt_with_cli(prompt: str, tokenizer_path: Path) -> Optional[List[int]]:
    spm_encode = shutil.which("spm_encode")
    if spm_encode is None or not tokenizer_path.exists():
        return None

    result = subprocess.run(
        [spm_encode, f"--model={tokenizer_path}", "--output_format=id"],
        input=f"{prompt}\n",
        text=True,
        capture_output=True,
        check=True,
    )
    output = result.stdout.strip()
    if not output:
        return []
    return [int(part) for part in output.split()]


def encode_prompt(prompt: str,
                  tokenizer_path: Path,
                  explicit_token_ids: Optional[str]) -> List[int]:
    if explicit_token_ids:
        return parse_token_ids(explicit_token_ids)

    tokenizer = try_load_sentencepiece(tokenizer_path)
    token_ids: Optional[List[int]] = None
    if tokenizer is not None:
        token_ids = tokenizer.encode(prompt)
    else:
        token_ids = encode_prompt_with_cli(prompt, tokenizer_path)

    if token_ids is None:
        raise RuntimeError(
            "sentencepiece is unavailable or tokenizer.model is missing. "
            "Install sentencepiece, make sure spm_encode is in PATH, or pass --token-ids."
        )

    if not token_ids:
        raise ValueError("prompt encoded to an empty token sequence")
    return token_ids


def decode_tokens(token_ids: List[int], tokenizer_path: Path) -> str:
    tokenizer = try_load_sentencepiece(tokenizer_path)
    if tokenizer is None:
        spm_decode = shutil.which("spm_decode")
        if spm_decode is not None and tokenizer_path.exists():
            result = subprocess.run(
                [spm_decode, f"--model={tokenizer_path}", "--input_format=id"],
                input=" ".join(str(token_id) for token_id in token_ids) + "\n",
                text=True,
                capture_output=True,
                check=True,
            )
            return result.stdout.strip()
        return " ".join(str(token_id) for token_id in token_ids)
    return tokenizer.decode(token_ids)


def dump_tokenizer_golden(prompt: str, token_ids: List[int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "# sentencepiece tokenizer golden\n"
        f"prompt={prompt}\n"
        "ids=" + " ".join(str(token_id) for token_id in token_ids) + "\n",
        encoding="utf-8",
    )


def format_float_values(values: List[float]) -> str:
    return " ".join(f"{value:.9g}" for value in values)


def dump_embedding_golden(prompt: str,
                          token_ids: List[int],
                          embedding_output,
                          output_path: Path) -> None:
    embedding_output = embedding_output.detach().cpu().to(dtype=embedding_output.dtype)
    shape = list(embedding_output.shape)
    flat_values = embedding_output.reshape(-1).to(dtype=embedding_output.dtype).tolist()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "# llama2 embedding golden\n"
        f"prompt={prompt}\n"
        "ids=" + " ".join(str(token_id) for token_id in token_ids) + "\n"
        f"dtype={embedding_output.dtype}\n"
        "shape=" + " ".join(str(dim) for dim in shape) + "\n"
        "values=" + format_float_values(flat_values) + "\n",
        encoding="utf-8",
    )


def build_device(device_arg: Optional[str]):
    import torch

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
    parser.add_argument("--dump-golden", dest="dump_golden", action="store_true",
                        help="Dump golden outputs and exit; currently includes tokenizer and embedding")
    parser.add_argument("--no-dump-golden", dest="dump_golden", action="store_false",
                        help="Disable golden dump mode and run generation instead")
    parser.set_defaults(dump_golden=True)
    parser.add_argument("--tokenizer-golden-output", type=Path, default=DEFAULT_TOKENIZER_GOLDEN,
                        help="Output path for tokenizer golden dump")
    parser.add_argument("--embedding-golden-output", type=Path, default=DEFAULT_EMBEDDING_GOLDEN,
                        help="Output path for embedding golden dump")
    args = parser.parse_args()

    prompt_token_ids = encode_prompt(args.prompt, args.tokenizer, args.token_ids)

    if args.dump_golden:
        dump_tokenizer_golden(args.prompt, prompt_token_ids, args.tokenizer_golden_output)
        print(f"tokenizer golden file: {args.tokenizer_golden_output}")
        print(f"prompt: {args.prompt}")
        print(f"token ids: {prompt_token_ids}")
        
        if not args.checkpoint.exists():
            raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

        import torch
        from load import load_checkpoint  # noqa: E402

        device = build_device(args.device)
        model = load_checkpoint(str(args.checkpoint))
        model = model.to(device)
        model.eval()

        input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)

        with torch.inference_mode():
            # In eval mode dropout is disabled, so this matches the tensor fed to layer 0.
            embedding_output = model.dropout(model.tok_embeddings(input_ids))

        dump_embedding_golden(
            args.prompt,
            prompt_token_ids,
            embedding_output,
            args.embedding_golden_output,
        )
        print(f"embedding golden file: {args.embedding_golden_output}")
        print(f"prompt: {args.prompt}")
        print(f"token ids: {prompt_token_ids}")
        print(f"embedding shape: {list(embedding_output.shape)}")
        return

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    import torch
    from load import load_checkpoint  # noqa: E402

    device = build_device(args.device)
    model = load_checkpoint(str(args.checkpoint))
    model = model.to(device)
    model.eval()

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
