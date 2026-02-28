import argparse
import asyncio
import csv
import io
import time
from dataclasses import dataclass

import httpx


@dataclass
class FetchResult:
    label: str
    url: str
    ok: bool
    status_code: int | None
    elapsed_s: float
    size_bytes: int | None = None
    error: str | None = None


def _bia_ftp_base_url(accession: str) -> str:
    accession_number = int(accession.split("S-BIAD", 1)[1])
    suffix = accession_number % 1000
    return f"https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/{suffix}/{accession}"


def _extract_tsv_pairs(tsv_content: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    reader = csv.DictReader(io.StringIO(tsv_content), delimiter="\t")
    for row in reader:
        label_path = str(row.get("Files") or "").strip()
        source_image = str(row.get("Source image") or "").strip()
        if not label_path or not source_image:
            continue
        pairs.append((source_image, label_path))
    return pairs


async def _timed_get(client: httpx.AsyncClient, label: str, url: str) -> FetchResult:
    started = time.perf_counter()
    try:
        resp = await client.get(url)
        elapsed = time.perf_counter() - started
        return FetchResult(
            label=label,
            url=url,
            ok=200 <= resp.status_code < 300,
            status_code=resp.status_code,
            elapsed_s=elapsed,
            size_bytes=len(resp.content),
            error=None if 200 <= resp.status_code < 300 else f"HTTP {resp.status_code}",
        )
    except Exception as exp:
        elapsed = time.perf_counter() - started
        return FetchResult(
            label=label,
            url=url,
            ok=False,
            status_code=None,
            elapsed_s=elapsed,
            size_bytes=None,
            error=str(exp),
        )


def _print_result(result: FetchResult) -> None:
    status = result.status_code if result.status_code is not None else "ERR"
    size = result.size_bytes if result.size_bytes is not None else "-"
    msg = f"[{result.label}] ok={result.ok} status={status} time={result.elapsed_s:.2f}s size={size}"
    if result.error:
        msg += f" error={result.error}"
    print(msg, flush=True)


async def main() -> None:
    parser = argparse.ArgumentParser(description="BIA-only fetch probe (no training calls).")
    parser.add_argument("--accession", default="S-BIAD1392")
    parser.add_argument("--sample-pairs", type=int, default=3)
    args = parser.parse_args()

    accession = str(args.accession).upper()
    base_url = _bia_ftp_base_url(accession)

    study_url = f"https://www.ebi.ac.uk/biostudies/bioimages/studies/{accession}"
    tsv_candidates = [
        f"{base_url}/Files/ps_ovule_labels.tsv",
        f"{base_url}/Files/labels.tsv",
        f"{base_url}/Files/{accession}_labels.tsv",
    ]

    timeout = httpx.Timeout(connect=20.0, read=120.0, write=20.0, pool=20.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        print(f"Probing accession: {accession}")

        study_result = await _timed_get(client, "study_page", study_url)
        _print_result(study_result)

        tsv_result: FetchResult | None = None
        tsv_text: str | None = None
        for tsv_url in tsv_candidates:
            result = await _timed_get(client, "tsv", tsv_url)
            _print_result(result)
            if result.ok:
                resp = await client.get(tsv_url)
                tsv_text = resp.text
                tsv_result = result
                break

        if not tsv_result or not tsv_text:
            print("No TSV source succeeded. BIA pair listing is blocked/slow at source.")
            return

        pairs = _extract_tsv_pairs(tsv_text)
        print(f"Parsed TSV pairs: {len(pairs)}")
        if not pairs:
            print("TSV loaded but contained no usable image/mask pairs.")
            return

        sample_count = max(1, min(int(args.sample_pairs), len(pairs)))
        sample_pairs = pairs[:sample_count]

        urls: list[tuple[str, str]] = []
        for idx, (image_rel, mask_rel) in enumerate(sample_pairs, start=1):
            image_url = f"{base_url}/Files/{image_rel.lstrip('/')}"
            mask_url = f"{base_url}/Files/{mask_rel.lstrip('/')}"
            urls.append((f"pair{idx}_image", image_url))
            urls.append((f"pair{idx}_mask", mask_url))

        print(f"Fetching {len(urls)} sample files from FTP...")
        results = await asyncio.gather(*[_timed_get(client, label, url) for label, url in urls])
        for result in results:
            _print_result(result)

        ok_count = sum(1 for item in results if item.ok)
        print(f"Sample fetch success: {ok_count}/{len(results)}")


if __name__ == "__main__":
    asyncio.run(main())
