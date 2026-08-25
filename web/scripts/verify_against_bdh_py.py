"""Check the TypeScript BDH toy against the reference PyTorch model in ../../bdh.py.

Development-only helper. It imports the repo's `bdh` module read-only, copies
weights exported by `npm run toy:export-ref` into a `bdh.BDH` configured with
n_head=1 and dropout=0, and compares logits with the TypeScript implementation
used by the web demo.

    pip install -r ../../requirements.txt
    npm run toy:export-ref
    python3 scripts/verify_against_bdh_py.py

Requires torch; it is not needed to build or deploy the site.
"""

import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO_ROOT)

import bdh  # noqa: E402  (path set up above)

CASE_PATH = os.path.join(HERE, ".cache", "reference-case.json")


def build_model(cfg):
    d = cfg["d"]
    n = cfg["n"]
    # bdh.py derives N = mlp_internal_dim_multiplier * n_embd // n_head.
    # With one head, N == n, so the multiplier is n / d.
    multiplier, remainder = divmod(n, d)
    assert remainder == 0, "toy n must be a multiple of d"
    config = bdh.BDHConfig(
        n_layer=cfg["layers"],
        n_embd=d,
        dropout=0.0,
        n_head=1,
        mlp_internal_dim_multiplier=multiplier,
        vocab_size=cfg["vocab"],
    )
    model = bdh.BDH(config)
    model.eval()
    return model, config


def load_weights(model, cfg, params):
    d, n, vocab = cfg["d"], cfg["n"], cfg["vocab"]

    def tensor(values, shape):
        return torch.tensor(values, dtype=torch.float32).reshape(shape)

    with torch.no_grad():
        # paper Dx  <-> bdh.py encoder   (nh, D, N)
        model.encoder.copy_(tensor(params["Dx"], (1, d, n)))
        # paper Dy  <-> bdh.py encoder_v (nh, D, N)
        model.encoder_v.copy_(tensor(params["Dy"], (1, d, n)))
        # paper E   <-> bdh.py decoder   (nh * N, D)
        model.decoder.copy_(tensor(params["E"], (n, d)))
        model.embed.weight.copy_(tensor(params["embed"], (vocab, d)))
        model.lm_head.copy_(tensor(params["head"], (d, vocab)))


def main():
    if not os.path.exists(CASE_PATH):
        raise SystemExit(f"missing {CASE_PATH}; run `npm run toy:export-ref` first")

    with open(CASE_PATH) as f:
        cases = json.load(f)

    print(f"torch {torch.__version__}, reference implementation {bdh.__file__}")
    failures = []

    for case in cases:
        cfg = case["config"]
        model, config = build_model(cfg)
        load_weights(model, cfg, case["params"])

        idx = torch.tensor(case["tokens"], dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(idx)
        torch_logits = logits.reshape(-1)

        scale = torch_logits.abs().max().item()
        report = []
        for label in ("parallel", "recurrent"):
            ts_logits = torch.tensor(case[f"logits_{label}"], dtype=torch.float32)
            max_abs = (torch_logits - ts_logits).abs().max().item()
            relative = max_abs / scale
            report.append((label, max_abs, relative))
            if relative > 1e-4:
                failures.append((case["name"], label, relative))

        params = sum(p.numel() for p in model.parameters())
        print(
            f'\ncase "{case["name"]}": BDH-GPU(n={cfg["n"]}, d={cfg["d"]}) '
            f'L={cfg["layers"]} T={len(case["tokens"])}'
        )
        print(f"  torch params {params:,} (includes the unused lm_gate)")
        print(f"  largest |logit| {scale:.6e}")
        for label, max_abs, relative in report:
            print(f"  {label:>9} vs bdh.py: max abs {max_abs:.3e}, relative {relative:.3e}")

    if failures:
        for name, label, relative in failures:
            print(f"FAIL {name}/{label}: relative difference {relative:.3e}")
        raise SystemExit(1)
    print("\nPASS: TypeScript port matches bdh.py to float32 precision")


if __name__ == "__main__":
    main()
