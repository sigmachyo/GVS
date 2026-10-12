import json
import plotly.graph_objects as go
import pandas as pd
import sys
import argparse
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "temp"

def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results for Work 3")
    parser.add_argument("json_file", help="Google Benchmark JSON output file")
    args = parser.parse_args()

    with open(args.json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    benchmarks = data.get("benchmarks", [])
    if not benchmarks:
        print("No benchmarks found in JSON")
        return

    eigen_data = {"N": [], "Time": []}
    shmem_data = {"N": [], "Time": []}
    naive_data = {"N": [], "Time": []}

    for b in benchmarks:
        name = b["name"]
        unit = b.get("time_unit", "ns")

        def to_ms(value):
            if unit == "ns":  return value / 1e6
            if unit == "us":  return value / 1e3
            if unit == "ms":  return value
            if unit == "s":   return value * 1e3
            return value

        time_val = to_ms(b["real_time"])

        if "BM_EigenMatrixMultiplication" in name:
            try:
                n = int(name.split("/")[1])
                eigen_data["N"].append(n)
                eigen_data["Time"].append(time_val)
            except:
                pass
        elif "BM_CUDAMatrixMultiplication_Shmem" in name:
            try:
                n = int(name.split("/")[1])
                shmem_data["N"].append(n)
                shmem_data["Time"].append(time_val)
            except:
                pass
        elif "BM_CUDAMatrixMultiplication_Naive" in name:
            try:
                n = int(name.split("/")[1])
                naive_data["N"].append(n)
                naive_data["Time"].append(time_val)
            except:
                pass

    df_eigen = pd.DataFrame(eigen_data).sort_values("N")
    df_shmem = pd.DataFrame(shmem_data).sort_values("N")
    df_naive = pd.DataFrame(naive_data).sort_values("N")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. График реальной вычислительной сложности (Real Complexity)
    fig_complexity = go.Figure()
    if not df_eigen.empty:
        fig_complexity.add_trace(go.Scatter(
            x=df_eigen["N"], y=df_eigen["Time"], 
            mode='lines+markers', name='Eigen (CPU)'
        ))
    if not df_naive.empty:
        fig_complexity.add_trace(go.Scatter(
            x=df_naive["N"], y=df_naive["Time"], 
            mode='lines+markers', name='CUDA Naive (предыдущая версия)'
        ))
    if not df_shmem.empty:
        fig_complexity.add_trace(go.Scatter(
            x=df_shmem["N"], y=df_shmem["Time"], 
            mode='lines+markers', name='CUDA Shared Memory (новая версия)'
        ))

    fig_complexity.update_layout(
        title="Реальная вычислительная сложность умножения матриц (Real Complexity)",
        xaxis_title="Размер матрицы N (N × N)",
        yaxis_title="Время выполнения, мс",
        xaxis_type="log",
        yaxis_type="log",
        template="plotly_white"
    )
    fig_complexity.write_html(OUTPUT_DIR / "real_complexity.html")
    try:
        fig_complexity.write_image(OUTPUT_DIR / "real_complexity.png")
    except Exception as e:
        print(f"Warning: could not export PNG image: {e}")

    # 2. График ускорения (Speedup)
    if not df_shmem.empty:
        fig_speedup = go.Figure()

        if not df_naive.empty:
            merged_naive = pd.merge(df_naive, df_shmem, on="N", suffixes=('_naive', '_shmem'))
            merged_naive["Speedup"] = merged_naive["Time_naive"] / merged_naive["Time_shmem"]
            fig_speedup.add_trace(go.Scatter(
                x=merged_naive["N"], y=merged_naive["Speedup"],
                mode='lines+markers', name='Ускорение относительно предыдущей версии (Naive GPU)'
            ))

        if not df_eigen.empty:
            merged_eigen = pd.merge(df_eigen, df_shmem, on="N", suffixes=('_eigen', '_shmem'))
            merged_eigen["Speedup"] = merged_eigen["Time_eigen"] / merged_eigen["Time_shmem"]
            fig_speedup.add_trace(go.Scatter(
                x=merged_eigen["N"], y=merged_eigen["Speedup"],
                mode='lines+markers', name='Ускорение относительно Eigen (CPU)'
            ))

        fig_speedup.update_layout(
            title="Ускорение блочной версии с разделяемой памятью (Speedup)",
            xaxis_title="Размер матрицы N (N × N)",
            yaxis_title="Коэффициент ускорения",
            xaxis_type="log",
            yaxis_type="log",
            template="plotly_white"
        )
        fig_speedup.write_html(OUTPUT_DIR / "speedup.html")
        try:
            fig_speedup.write_image(OUTPUT_DIR / "speedup.png")
        except Exception as e:
            print(f"Warning: could not export PNG image: {e}")

    print("Charts generated successfully in", OUTPUT_DIR)

if __name__ == "__main__":
    main()
