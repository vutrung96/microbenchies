import csv
import statistics

import torch
import torch.utils.benchmark as benchmark


def separate_matmuls(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Two separate matmuls: X @ W1 and X @ W2."""
    return x @ w1, x @ w2


def fused_matmul(x: torch.Tensor, w_fused: torch.Tensor, f: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Single fused matmul: X @ concat(W1, W2), then split."""
    out = x @ w_fused
    return out[:, :f], out[:, f:]


def benchmark_matmuls(b: int, d: int, f: int, device: str = "cuda", dtype: torch.dtype = torch.float16):
    """Benchmark separate vs fused matmuls."""
    x = torch.randn(b, d, device=device, dtype=dtype)
    w1 = torch.randn(d, f, device=device, dtype=dtype)
    w2 = torch.randn(d, f, device=device, dtype=dtype)
    w_fused = torch.cat([w1, w2], dim=1)  # Shape: D x 2F

    # Warmup
    for _ in range(10):
        separate_matmuls(x, w1, w2)
        fused_matmul(x, w_fused, f)
    torch.cuda.synchronize()

    dtype_name = str(dtype).split(".")[-1]

    # Benchmark separate matmuls
    t_separate = benchmark.Timer(
        stmt="separate_matmuls(x, w1, w2)",
        globals={"separate_matmuls": separate_matmuls, "x": x, "w1": w1, "w2": w2},
        label="matmul",
        sub_label=f"{dtype_name} B={b}, D={d}, F={f}",
        description="separate",
    )

    # Benchmark fused matmul
    t_fused = benchmark.Timer(
        stmt="fused_matmul(x, w_fused, f)",
        globals={"fused_matmul": fused_matmul, "x": x, "w_fused": w_fused, "f": f},
        label="matmul",
        sub_label=f"{dtype_name} B={b}, D={d}, F={f}",
        description="fused",
    )

    return t_separate.blocked_autorange(min_run_time=1.0), t_fused.blocked_autorange(min_run_time=1.0)


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, running on CPU")
        device = "cpu"
    else:
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name()}")

    # Test configurations (B=batch, D=input dim, F=output dim per weight)
    batch_sizes = [1, 32, 128, 512, 1024]
    model_configs = [
        (4096, 14336),   # Llama-like dims
        (8192, 28672),   # Larger model
    ]
    dtypes = [torch.float16, torch.bfloat16, torch.float32]
    configs = [(b, d, f, dt) for dt in dtypes for d, f in model_configs for b in batch_sizes]

    results = []
    csv_rows = []
    for b, d, f, dt in configs:
        dtype_name = str(dt).split(".")[-1]
        print(f"\nBenchmarking B={b}, D={d}, F={f}, dtype={dtype_name}...")
        try:
            r_sep, r_fused = benchmark_matmuls(b, d, f, device=device, dtype=dt)
            results.extend([r_sep, r_fused])
            gpu_name = torch.cuda.get_device_name() if device == "cuda" else "CPU"
            sep_std = statistics.stdev(r_sep.times) * 1e6 if len(r_sep.times) > 1 else 0
            fused_std = statistics.stdev(r_fused.times) * 1e6 if len(r_fused.times) > 1 else 0
            csv_rows.append({
                "accelerator": gpu_name,
                "dtype": dtype_name,
                "B": b,
                "D": d,
                "F": f,
                "separate_mean_us": r_sep.mean * 1e6,
                "separate_std_us": sep_std,
                "separate_runs": len(r_sep.times),
                "fused_mean_us": r_fused.mean * 1e6,
                "fused_std_us": fused_std,
                "fused_runs": len(r_fused.times),
                "speedup": r_sep.mean / r_fused.mean,
            })
        except torch.cuda.OutOfMemoryError:
            print("  Skipped (OOM)")
            continue

    # Print comparison table
    compare = benchmark.Compare(results)
    compare.print()

    # Write to CSV
    if csv_rows:
        with open("fused_matmul.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print("\nResults written to fused_matmul.csv")


if __name__ == "__main__":
    main()
