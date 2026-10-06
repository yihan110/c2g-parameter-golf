"""
C2G eval helper (ruanyihan).

Standalone utilities for the Parameter Golf challenge:
  1. compute_bpb(...)        - tokenizer-agnostic Bits-Per-Byte (mirrors OpenAI train_gpt.py logic)
  2. welch_ttest(...)        - Welch's t-test for comparing 3-seed BPB means vs baseline
  3. CLI:  python ruanyihan_C2G_eval_bpb.py <val1> <val2> <val3> --baseline 1.2244

This script does NOT train anything; it only helps you verify the final number and
the statistical significance required by the leaderboard (delta >= 0.005, p < 0.01).
It requires torch + sentencepiece to run on the GPU box (or on the machine with data).

Usage for significance check (no GPU needed):
  python ruanyihan_C2G_eval_bpb.py 1.1701 1.1698 1.1705 --baseline 1.2244
"""

from __future__ import annotations

import argparse
import math
from statistics import mean, stdev
from typing import Sequence


def welch_ttest(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Two-sample Welch's t-test. Returns (t_stat, p_value_two_tailed)."""
    na, nb = len(a), len(b)
    ma, mb = mean(a), mean(b)
    va = (stdev(a) ** 2) if na > 1 else 0.0
    vb = (stdev(b) ** 2) if nb > 1 else 0.0
    denom = math.sqrt(va / na + vb / nb)
    if denom == 0.0:
        return 0.0, 1.0
    t = (ma - mb) / denom
    # Degrees of freedom (Welch-Satterthwaite)
    num = (va / na + vb / nb) ** 2
    den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1) if (na > 1 and nb > 1) else 1.0
    df = num / den if den > 0 else 1.0
    # Two-tailed p via Student-t CDF approximation (survival function)
    p = _student_t_sf(abs(t), df) * 2.0
    return t, p


def _student_t_sf(x: float, df: float) -> float:
    """Survival function P(T > x) for Student-t with df degrees of freedom (regularized incomplete beta)."""
    # Use scipy if available for exactness; else a light numeric fallback.
    try:
        from scipy import stats  # type: ignore

        return float(stats.t.sf(x, df))
    except Exception:
        # Euler-Maclaurin trapezoid integral of the t pdf from x to +inf.
        import math as _m

        steps = 20000
        h = 10.0 / steps  # integrate up to +10
        lo = x
        acc = 0.0

        def pdf(u: float) -> float:
            return (
                _m.gamma((df + 1) / 2.0)
                / (_m.sqrt(df * _m.pi) * _m.gamma(df / 2.0))
                * (1 + u * u / df) ** (-(df + 1) / 2.0)
            )

        acc += 0.5 * pdf(lo)
        for i in range(1, steps):
            acc += pdf(lo + i * h)
        acc += 0.5 * pdf(lo + steps * h)
        return acc * h


def compute_bpb(
    model,
    val_tokens,
    seq_len: int,
    base_bytes_lut,
    has_leading_space_lut,
    is_boundary_token_lut,
    device="cuda",
) -> float:
    """
    Compute tokenizer-agnostic BPB over a token stream, mirroring OpenAI's eval.
    model(x, y) must return mean CE loss. Returns val_bpb.
    """
    import torch

    model.eval()
    with torch.inference_mode():
        usable = ((val_tokens.numel() - 1) // seq_len) * seq_len
        tokens = val_tokens[: usable + 1]
        n_seqs = usable // seq_len
        loss_sum = 0.0
        token_count = 0
        byte_count = 0.0
        for i in range(n_seqs):
            local = tokens[i * seq_len : (i + 1) * seq_len + 1].to(device, dtype=torch.int64)
            x = local[:-1].reshape(1, seq_len)
            y = local[1:].reshape(1, seq_len)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=True):
                loss = model(x, y).detach().float()
            loss_sum += float(loss) * seq_len
            token_count += seq_len
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            byte_count += float(token_bytes.float().sum())
        val_loss = loss_sum / max(token_count, 1)
        bits_per_token = val_loss / math.log(2.0)
        tokens_per_byte = token_count / max(byte_count, 1e-9)
    model.train()
    return float(bits_per_token * tokens_per_byte)


def main() -> None:
    ap = argparse.ArgumentParser(description="C2G BPB significance checker")
    ap.add_argument("seed_values", nargs="+", type=float, help="3 seed val_bpb values, e.g. 1.1701 1.1698 1.1705")
    ap.add_argument("--baseline", type=float, default=1.2244, help="baseline BPB to compare against")
    ap.add_argument("--n-baseline-seeds", type=int, default=3, help="seeds behind the baseline mean")
    ap.add_argument("--min-delta", type=float, default=0.005, help="leaderboard min improvement (nats)")
    args = ap.parse_args()

    ours = args.seed_values
    if len(ours) < 3:
        raise SystemExit("Provide at least 3 seed values.")
    mean_ours = mean(ours)
    std_ours = stdev(ours) if len(ours) > 1 else 0.0
    delta = args.baseline - mean_ours

    # Treat baseline as a fixed reference (mean over its own seeds, sd unknown -> use ours as sd estimate)
    baseline_seeds = [args.baseline] * args.n_baseline_seeds
    t, p = welch_ttest(ours, baseline_seeds)

    print(f"ours:      mean={mean_ours:.6f} std={std_ours:.6f} n={len(ours)}")
    print(f"baseline:  {args.baseline:.6f}")
    print(f"delta:     {delta:+.6f}  (need >= {args.min_delta:.3f})")
    print(f"Welch t:   {t:.3f}   p={p:.5f}   (need p<0.01)")
    print("RESULT: " + ("PASS (statistically significant win)" if (delta >= args.min_delta and p < 0.01) else "FAIL (not significant)"))


if __name__ == "__main__":
    main()
