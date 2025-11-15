import argparse
import time
from pathlib import Path


def load_text(path: Path) -> str:
  with path.open("r", encoding="utf-8", errors="ignore") as f:
    return f.read()


def load_lines(path: Path) -> list[str]:
  with path.open("r", encoding="utf-8", errors="ignore") as f:
    return [line.rstrip("\n") for line in f]


def load_gpu_tokens(path: Path) -> list[list[str]]:
  if not path.exists():
    print(f"GPU token file not found: {path}")
    return []
  lines: list[list[str]] = []
  with path.open("r", encoding="utf-8", errors="ignore") as f:
    for line in f:
      line = line.strip()
      if not line:
        lines.append([])
      else:
        lines.append(line.split())
  return lines


def write_tokens_file(path: Path, seqs: list[list[str]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
    for line in seqs:
      if line:
        f.write(" ".join(line))
      f.write("\n")
  print(f"Wrote tokenized text to {path}")


def bench_tiktoken(text: str):
  try:
    import tiktoken  # type: ignore
  except ImportError:
    print("tiktoken not installed; run: pip install --user tiktoken")
    return None

  enc = tiktoken.get_encoding("gpt2")
  t0 = time.time()
  tokens = enc.encode(text)
  t1 = time.time()
  dt = t1 - t0
  tok_count = len(tokens)
  tok_per_s = tok_count / dt if dt > 0 else float("inf")
  print(f"[tiktoken gpt2] tokens: {tok_count}, time: {dt*1000:.3f} ms, throughput: {tok_per_s:.1f} tokens/s")
  return enc


def bench_hf_gpt2(text: str):
  try:
    from transformers import AutoTokenizer  # type: ignore
  except ImportError:
    print("transformers not installed; run: pip install --user transformers")
    return None

  tok = AutoTokenizer.from_pretrained("gpt2")
  # HF tokenizers typically operate on full text; batch_size=1 here to keep it simple
  t0 = time.time()
  out = tok(text, return_attention_mask=False, return_token_type_ids=False)
  t1 = time.time()
  input_ids = out["input_ids"]
  if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
    tok_count = sum(len(seq) for seq in input_ids)
  else:
    tok_count = len(input_ids)
  dt = t1 - t0
  tok_per_s = tok_count / dt if dt > 0 else float("inf")
  print(f"[HF GPT2Tokenizer] tokens: {tok_count}, time: {dt*1000:.3f} ms, throughput: {tok_per_s:.1f} tokens/s")
  return tok


def normalize_bpe_symbol(tok: str) -> str:
  # Map GPT-2 style "ĠProject" to the literal substring " Project"
  if tok.startswith("Ġ"):
    return " " + tok[1:]
  return tok


def compare_segmentations_full(text: str,
                               gpu_tokens: list[list[str]],
                               enc,
                               hf_tok) -> None:
  # Flatten GPU tokens across chunks into a single sequence
  flat_gpu: list[str] = [t for line in gpu_tokens for t in line]
  if not flat_gpu:
    print("No GPU tokens loaded; skipping correctness comparison.")
    return

  # Normalize GPU tokens to decoded-substring style
  norm_gpu = [normalize_bpe_symbol(t) for t in flat_gpu]

  # Build tiktoken token sequence for full text
  tk_seq: list[str] = []
  if enc is not None:
    ids = enc.encode(text)
    tk_seq = [enc.decode_single_token_bytes(t).decode("utf-8", errors="replace") for t in ids]

  # Build HF GPT2 token sequence for full text
  hf_seq: list[str] = []
  if hf_tok is not None:
    hf_raw = hf_tok.tokenize(text)
    hf_seq = [normalize_bpe_symbol(t) for t in hf_raw]

  def stats(name: str, ref_seq: list[str]):
    if not ref_seq:
      print(f"{name}: tokenizer not available; skipping.")
      return
    a = ref_seq
    b = norm_gpu
    len_diff = abs(len(a) - len(b))
    L = max(len(a), len(b))
    if L == 0:
      print(f"{name}: empty sequences; nothing to compare.")
      return
    matches = sum(1 for j in range(min(len(a), len(b))) if a[j] == b[j])
    match_ratio = matches / L
    print(f"[{name} vs GPU] |len_diff|: {len_diff}, positional token match: {match_ratio*100:.2f}%")

  stats("tiktoken gpt2", tk_seq)
  stats("HF GPT2Tokenizer", hf_seq)

  # Write out tokenized text for visual inspection
  out_dir = Path("data/output")
  if tk_seq:
    write_tokens_file(out_dir / "tiktoken_tokens.txt", [tk_seq])
  if hf_seq:
    write_tokens_file(out_dir / "hf_gpt2_tokens.txt", [hf_seq])


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--input", type=str, default="data/input/pride_and_prejudice.txt",
                  help="Path to input text file")
  ap.add_argument("--gpu_tokens", type=str, default="data/output/gpu_tokens.txt",
                  help="Path to GPU tokenizer output (token strings per line)")
  args = ap.parse_args()

  input_path = Path(args.input)
  if not input_path.exists():
    print(f"Input file not found: {input_path}")
    return

  text = load_text(input_path)
  gpu_tokens = load_gpu_tokens(Path(args.gpu_tokens))

  print(f"Loaded {len(text)} characters from {input_path}")

  enc = bench_tiktoken(text)
  hf_tok = bench_hf_gpt2(text)

  compare_segmentations_full(text, gpu_tokens, enc, hf_tok)


if __name__ == "__main__":
  main()


