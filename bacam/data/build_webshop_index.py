#!/usr/bin/env python3
"""Build the reusable GiGPO 1k-product WebShop Lucene index."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from bacam.paths import PATHS


VERL_AGENT_COMMIT = "35b3da38293993f9bf4f7873dfb3262a361e956c"
EXPECTED_SHA256 = {
    "items_shuffle_1000.json": "30a4765c3a327af72d9a9a95a6b2486d516f0fa1d3ecd83681901ce82a21b269",
    "items_ins_v2_1000.json": "f88a36314a397b53b3d9c3fa5878e5f7b26d35019a51ec83fbedeca61a948f6f",
    "items_human_ins.json": "cf78667548a71786e1d9049c24b802e48e1084ad4bb021cae56ce1f6d96954a3",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def verify_inputs(data_root: Path) -> dict[str, str]:
    actual = {}
    for name, expected in EXPECTED_SHA256.items():
        path = data_root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual[name] = sha256(path)
        if actual[name] != expected:
            raise RuntimeError(
                f"WebShop input drift for {path}: expected {expected}, got {actual[name]}")
    return actual


def write_documents(data_root: Path, webshop_root: Path, target: Path) -> int:
    sys.path.insert(0, str(webshop_root))
    from web_agent_site.engine import engine

    engine.HUMAN_ATTR_PATH = str(data_root / "items_human_ins.json")
    products, *_ = engine.load_products(
        filepath=str(data_root / "items_shuffle_1000.json"),
        attrpath=str(data_root / "items_ins_v2_1000.json"),
        human_goals=False,
    )
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for product in tqdm(products, desc="WebShop Lucene documents"):
            option_texts = []
            for option_name, values in product.get("options", {}).items():
                option_texts.append(f"{option_name}: {', '.join(values)}")
            document = {
                "id": product["asin"],
                "contents": " ".join([
                    product["Title"],
                    product["Description"],
                    product["BulletPoints"][0],
                    ", and ".join(option_texts),
                ]).lower(),
                "product": product,
            }
            handle.write(json.dumps(document) + "\n")
    os.replace(temporary, target)
    return len(products)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path,
                        default=Path(PATHS["BACAM_DATA_ROOT"]) / "webshop")
    parser.add_argument(
        "--webshop-root", type=Path,
        default=Path(PATHS["BACAM_VERL_AGENT_ROOT"]) / "agent_system/"
                     "environments/env_package/webshop/webshop",
    )
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    webshop_root = args.webshop_root.resolve()
    output_root = data_root / "search_engine_1k"
    resource_dir = output_root / "resources"
    documents = resource_dir / "documents.jsonl"
    resource_manifest = output_root / "documents_manifest.json"
    index = output_root / "indexes"
    index_building = output_root / "indexes.building"
    index_manifest = output_root / "index_manifest.json"
    output_root.mkdir(parents=True, exist_ok=True)
    resource_dir.mkdir(parents=True, exist_ok=True)

    hashes = verify_inputs(data_root)
    documents_ready = False
    if documents.is_file() and resource_manifest.is_file():
        previous = json.loads(resource_manifest.read_text())
        documents_ready = previous.get("input_sha256") == hashes
    if not documents_ready:
        count = write_documents(data_root, webshop_root, documents)
        atomic_json(resource_manifest, {
            "schema": "fm15-webshop-lucene-documents/v1",
            "verl_agent_commit": VERL_AGENT_COMMIT,
            "input_sha256": hashes,
            "documents": count,
            "path": str(documents),
            "bytes": documents.stat().st_size,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
    else:
        count = int(previous["documents"])

    if index.is_dir() and index_manifest.is_file():
        previous_index = json.loads(index_manifest.read_text())
        if previous_index.get("input_sha256") == hashes:
            if "indexed_documents" not in previous_index:
                from pyserini.search.lucene import LuceneSearcher
                existing_searcher = LuceneSearcher(str(index))
                indexed = int(existing_searcher.num_docs)
                del existing_searcher
                previous_index["indexed_documents"] = indexed
                previous_index["empty_documents"] = count - indexed
                atomic_json(index_manifest, previous_index)
            print(index_manifest.read_text())
            return
        raise RuntimeError(f"existing WebShop index has different inputs: {index}")

    if index_building.exists():
        shutil.rmtree(index_building)
    command = [
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", str(resource_dir),
        "--index", str(index_building),
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "1",
        "--storePositions", "--storeDocvectors", "--storeRaw",
    ]
    environment = os.environ.copy()
    environment["JAVA_HOME"] = str(Path(sys.prefix) / "lib/jvm")
    environment["PATH"] = f"{Path(sys.prefix) / 'bin'}:{environment['PATH']}"
    subprocess.run(command, check=True, env=environment)

    from pyserini.search.lucene import LuceneSearcher
    searcher = LuceneSearcher(str(index_building))
    indexed = int(searcher.num_docs)
    del searcher

    os.replace(index_building, index)
    index_files = [path for path in index.rglob("*") if path.is_file()]
    atomic_json(index_manifest, {
        "schema": "fm15-webshop-lucene-index/v1",
        "verl_agent_commit": VERL_AGENT_COMMIT,
        "input_sha256": hashes,
        "documents": count,
        "indexed_documents": indexed,
        "empty_documents": count - indexed,
        "index": str(index),
        "index_files": len(index_files),
        "index_bytes": sum(path.stat().st_size for path in index_files),
        "engine": "Pyserini 0.17.0 / Lucene BM25",
        "threads": 1,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    print(index_manifest.read_text())


if __name__ == "__main__":
    main()
