import json
import plotly.graph_objects as go
import pandas as pd
import sys
import argparse
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "temp"

def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("json_file", help="Google Benchmark JSON output file")
    args = parser.parse_args()

    with open(args.json_file, 'r') as f:
        data = json.load(f)

    benchmarks = data.get("benchmarks", [])
    if not benchmarks:
        print("No benchmarks found in JSON")
        return

    eigen_data = {"N": [], "Time": []}
    cuda_data = {"N": [], "Time": []}

    for b in benchmarks:
        name = b["name"]
        unit = b.get("time_unit", "ns")

        def to_ms(value):
            if unit == "ns":  return value / 1e6
            if unit == "us":  return value / 1e3
            if unit == "ms":  return value
            if unit == "s":   return value * 1e3

        if "BM_EigenMatrixMultiplication" in name:
            try:
                n = int(name.split("/")[1])
                eigen_data["N"].append(n)
                eigen_data["Time"].append(to_ms(b["real_time"]))
            except:
                pass
        elif "BM_CUDAMatrixMultiplication" in name:
            try:
                n = int(name.split("/")[1])
                cuda_data["N"].append(n)
                cuda_data["Time"].append(to_ms(b["real_time"]))
            except:
                pass

    df_eigen = pd.DataFrame(eigen_data).sort_values("N")
    df_cuda = pd.DataFrame(cuda_data).sort_values("N")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig_complexity = go.Figure()
    fig_complexity.add_trace(go.Scatter(x=df_eigen["N"], y=df_eigen["Time"], mode='lines+markers', name='Eigen Matrix Multiplication (CPU)'))
    fig_complexity.add_trace(go.Scatter(x=df_cuda["N"],  y=df_cuda["Time"],  mode='lines+markers', name='CUDA Matrix Multiplication (GPU)'))

    fig_complexity.update_layout(
        title="Real Complexity",
        xaxis_title="N (matrix size)",
        yaxis_title="Time, ms",
        xaxis_type="log",
        yaxis_type="log",
        template="plotly_white"
    )
    fig_complexity.write_html(OUTPUT_DIR / "real_complexity.html")
    try:
        fig_complexity.write_image(OUTPUT_DIR / "real_complexity.png")
    except Exception as e:
        print(f"Warning: could not export PNG image: {e}")

    if not df_eigen.empty and not df_cuda.empty:
        merged = pd.merge(df_eigen, df_cuda, on="N", suffixes=('_eigen', '_cuda'))
        merged["Speedup"] = merged["Time_eigen"] / merged["Time_cuda"]

        fig_speedup = go.Figure()
        fig_speedup.add_trace(go.Scatter(x=merged["N"], y=merged["Speedup"], mode='lines+markers', name='Speedup'))

        fig_speedup.update_layout(
            title="Speedup: CUDA Matrix Multiplication (GPU) vs Eigen (CPU)",
            xaxis_title="N (matrix size)",
            yaxis_title="Speedup",
            xaxis_type="log",
            yaxis_type="log",
            template="plotly_white"
        )
        fig_speedup.write_html(OUTPUT_DIR / "speedup.html")
        try:
            fig_speedup.write_image(OUTPUT_DIR / "speedup.png")
        except Exception as e:
            print(f"Warning: could not export PNG image: {e}")

if __name__ == "__main__":
    main()
