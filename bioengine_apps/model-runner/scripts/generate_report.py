import argparse
import base64
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

def generate_html_report(
    output_dir: Path,
    summary: Dict,
    figure_images: Dict[str, Dict[str, bytes]],
) -> Path:
    """Generate a self-contained single HTML report with summary, figures, and metric table."""
    report_path = output_dir / "model_comparison_report.html"

    original_prompt = summary.get("original_prompt", "N/A")
    task_description = summary.get("task", "Unknown Task")
    keywords = summary.get("keywords", [])
    metrics = summary.get("metrics", {})
    model_errors = summary.get("failed_models", {})
    candidates = summary.get("candidates", [])
    
    # Calculate successful_models and best items if not explicitly set
    candidate_ids = candidates if isinstance(candidates, list) else list(candidates)
    successful_ids = [m for m in candidate_ids if m not in model_errors]
    
    best_model_id = summary.get("best_model", {}).get("id", "N/A")
    best_model_metrics = summary.get("best_model", {}).get("metrics", {})

    sorted_metrics = sorted(
        metrics.items(),
        key=lambda item: item[1].get("iou", float("-inf")) if isinstance(item[1], dict) else float("-inf"),
        reverse=True,
    )

    def format_metric(value: float, decimals: int = 4) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return "inf" if value > 0 else "-inf"
        return f"{value:.{decimals}f}"


    metric_rows = []
    # Identify available metric keys dynamically, putting iou, dice, psnr first if available
    if sorted_metrics and isinstance(sorted_metrics[0][1], dict):
        all_metric_keys = list(sorted_metrics[0][1].keys())
    else:
        all_metric_keys = ["iou", "dice", "psnr"]
        
    prioritized_keys = [k for k in ["iou", "dice", "psnr"] if k in all_metric_keys]
    other_keys = [k for k in all_metric_keys if k not in prioritized_keys]
    display_keys = prioritized_keys + other_keys

    for rank, (model_id, vals) in enumerate(sorted_metrics, start=1):
        row_html = f"""
            <tr class="border-b border-gray-200 hover:bg-gray-50">
                <td class="px-4 py-3 text-sm text-gray-900">{rank}</td>
                <td class="px-4 py-3 text-sm font-medium text-gray-900">{model_id}</td>
        """
        for k in display_keys:
            val = vals.get(k, 0.0) if isinstance(vals, dict) else 0.0
            row_html += f'<td class="px-4 py-3 text-sm text-gray-800 text-right">{format_metric(val, 4)}</td>'
        row_html += "</tr>"
        metric_rows.append(row_html)

    header_html = """
        <tr>
            <th class="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-gray-700">Rank</th>
            <th class="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-gray-700">Model ID</th>
    """
    for k in display_keys:
        header_html += f'<th class="px-4 py-3 text-right text-xs font-semibold uppercase tracking-wide text-gray-700">{k.upper()}</th>'
    header_html += "</tr>"

    failed_rows = []
    for model_id, error_msg in model_errors.items():
        failed_rows.append(
            f"""
            <tr class="border-b border-gray-200">
                <td class="px-4 py-3 text-sm font-medium text-gray-900">{model_id}</td>
                <td class="px-4 py-3 text-sm text-gray-700">{error_msg[:280]}</td>
            </tr>
            """
        )

    figures_html = []
    for title, image_info in figure_images.items():
        filename = image_info["filename"]
        image_bytes = image_info["bytes"]
        
        ext = filename.lower().split('.')[-1]
        mime_type = "image/svg+xml" if ext == "svg" else "image/png"
        
        image_src = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
        figures_html.append(
            f"""
            <div class="mb-6 last:mb-0">
                <div class="flex items-center justify-between gap-3 mb-3">
                    <h3 class="text-base font-semibold text-gray-900">{title}</h3>
                    <button class="save-figure-btn px-3 py-1.5 text-sm font-medium rounded-md bg-blue-600 text-white hover:bg-blue-700" data-filename="{filename}" data-src="{image_src}">Export Figure</button>
                </div>
                <img src="{image_src}" alt="{title}" class="w-full rounded-lg border border-gray-200" />
            </div>
            """
        )

    summary_json = json.dumps(summary, indent=2, default=str)

    html = f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Model Comparison Report</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-50 text-gray-900">
    <main class="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
        <header class="mb-8">
            <h1 class="text-3xl font-bold tracking-tight">BioImage.IO Model Comparison Report</h1>
        </header>

        <section class="bg-white rounded-xl border border-gray-200 shadow-sm p-6 mb-6">
            <h2 class="text-xl font-semibold mb-4">Instruction</h2>
            <div class="mb-4">
                <h3 class="text-sm font-semibold text-gray-700 uppercase tracking-wide">Original Prompt</h3>
                <p class="text-sm text-gray-800 leading-6 mt-1 whitespace-pre-wrap">{original_prompt}</p>
            </div>
            <div class="mb-4">
                <h3 class="text-sm font-semibold text-gray-700 uppercase tracking-wide">Task</h3>
                <p class="text-sm text-gray-800 leading-6 mt-1">{task_description}</p>
            </div>
            <div>
                <h3 class="text-sm font-semibold text-gray-700 uppercase tracking-wide">Keywords</h3>
                <p class="text-sm text-gray-800 leading-6 mt-1">{', '.join(keywords) if keywords else 'N/A'}</p>
            </div>
        </section>

        <section class="bg-white rounded-xl border border-gray-200 shadow-sm p-6 mb-6">
            <h2 class="text-xl font-semibold mb-4">Summary</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div class="rounded-lg bg-gray-50 p-4 border border-gray-200">
                    <dt class="text-xs uppercase tracking-wide text-gray-500">Candidates / Successful</dt>
                    <dd class="mt-1 text-sm font-medium">{len(successful_ids)} / {len(candidate_ids)} successful</dd>
                </div>
                <div class="rounded-lg bg-gray-50 p-4 border border-gray-200">
                    <dt class="text-xs uppercase tracking-wide text-gray-500">Best Model: {best_model_id}</dt>
                    <dd class="mt-1 text-sm font-medium">
                        {'<br/>'.join([f"{k.upper()}: {format_metric(v)}" for k, v in best_model_metrics.items()]) if best_model_metrics else 'N/A'}
                    </dd>
                </div>
            </div>
        </section>

        <section class="bg-white rounded-xl border border-gray-200 shadow-sm p-6 mb-6">
            <h2 class="text-xl font-semibold mb-4">Generated Figures</h2>
            <div class="grid grid-cols-1 gap-4">
                {''.join(figures_html)}
            </div>
        </section>

        <section class="bg-white rounded-xl border border-gray-200 shadow-sm p-6 mb-6">
            <h2 class="text-xl font-semibold mb-4">Metrics Table</h2>
            <div class="overflow-x-auto">
                <table class="min-w-full border border-gray-200 rounded-lg overflow-hidden">
                    <thead class="bg-gray-100">
                        {header_html}
                    </thead>
                    <tbody>
                        {''.join(metric_rows) if metric_rows else '<tr><td colspan="5" class="px-4 py-3 text-sm text-gray-600">No successful model metrics available.</td></tr>'}
                    </tbody>
                </table>
            </div>
        </section>

        <section class="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
            <h2 class="text-xl font-semibold mb-4">Failed Models</h2>
            <div class="overflow-x-auto">
                <table class="min-w-full border border-gray-200 rounded-lg overflow-hidden">
                    <thead class="bg-gray-100">
                        <tr>
                            <th class="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-gray-700">Model ID</th>
                            <th class="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-gray-700">Error</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(failed_rows) if failed_rows else '<tr><td colspan="2" class="px-4 py-3 text-sm text-gray-600">No model failures.</td></tr>'}
                    </tbody>
                </table>
            </div>
        </section>
    </main>
        <script>
            function saveFigure(dataUri, filename) {{
                const a = document.createElement('a');
                a.href = dataUri;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            }}

            document.querySelectorAll('.save-figure-btn').forEach((btn) => {{
                btn.addEventListener('click', () => {{
                    saveFigure(btn.dataset.src, btn.dataset.filename);
                }});
            }});
        </script>
</body>
</html>
"""

    report_path.write_text(html, encoding="utf-8")
    print(f"  HTML report saved: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Generate BioImage.IO Model Comparison Report")
    parser.add_argument("--summary", required=True, help="Path to comparison_summary.json")
    parser.add_argument("--montage", required=True, help="Path to model_comparison_montage.png")
    parser.add_argument("--barplot", required=True, help="Path to model_comparison_barplot.png", type=str)
    parser.add_argument("--output-dir", default="./comparison_results", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.summary, "r") as f:
        summary = json.load(f)

    figure_images = {}
    
    with open(args.montage, "rb") as f:
        figure_images["Model Output Montage"] = {
            "filename": Path(args.montage).name,
            "bytes": f.read()
        }

    barplots = args.barplot.split(",")
    for bp in barplots:
        name = Path(bp).stem.replace("_", " ").title()
        with open(bp, "rb") as f:
            figure_images[name] = {
                "filename": Path(bp).name,
                "bytes": f.read()
            }

    generate_html_report(output_dir, summary, figure_images)


if __name__ == "__main__":
    main()
