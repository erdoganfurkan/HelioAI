"""Share a built index: export it to files, fetch a published one, import it.

Building the index walks the speasy inventory and embeds ~83 000 products — the better part
of an hour on a laptop, before a researcher can ask a first question. For a given catalogue
and embedding model the result is the same on every machine, so CI builds it once per
release (`.github/workflows/index.yml`) and publishes a snapshot on the Hugging Face Hub;
`helioai index` on an empty index fetches that snapshot instead of building.

A snapshot is the content of the collections — ids, documents, metadata, embeddings — and
not Chroma's directory: a store written by one Chroma release is not guaranteed to open
in an older one, and `chromadb` carries no upper bound. Importing re-creates the
collections through `open_collections`, with the HNSW settings a local build uses.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

FORMAT = 1
MANIFEST = "manifest.json"
MIN_KEPT = 0.95
_PAGE = 5000


def _collection_names() -> dict[str, str]:
    from helioai.config import settings

    return {
        "products": settings.rag.collection_name,
        "catalogs": settings.rag.catalogs_collection_name,
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _release_clients() -> None:
    from chromadb.api.client import SharedSystemClient

    SharedSystemClient.clear_system_cache()


def index_is_empty(chroma_dir: Path | None = None) -> bool:
    """Whether the product collection is absent or holds nothing.

    This is the condition under which `helioai index` fetches the published snapshot
    rather than walking the inventory, and a fetch replaces the index. So a store that
    exists but cannot be read — corrupt, or written by a Chroma this one cannot open — is
    not empty: `helioai index` then builds on it and fails loudly, rather than discarding
    products a user built or classified.
    """
    from helioai.config import settings

    chroma_dir = Path(chroma_dir or settings.rag.chroma_dir)
    if not (chroma_dir / "chroma.sqlite3").exists():
        return True
    import chromadb

    name = settings.rag.collection_name
    try:
        client = chromadb.PersistentClient(path=str(chroma_dir))
        listed = {c if isinstance(c, str) else c.name for c in client.list_collections()}
        return name not in listed or client.get_collection(name).count() == 0
    except Exception:
        return False
    finally:
        _release_clients()


def export_index(out_dir: Path, chroma_dir: Path | None = None) -> dict:
    """Write the index as a snapshot: per collection a gzipped JSONL and a float32 `.npy`.

    The records keep Chroma's order, so an import inserts in the order the build did. The
    manifest carries a SHA-256 per file — the import refuses a truncated download rather
    than serving a partial index — and the embedding model, since vectors from another
    model would be silently meaningless against this install's query embeddings.

    Args:
        out_dir: Destination directory, created when absent.
        chroma_dir: The index to export; `settings.rag.chroma_dir` by default.

    Returns:
        The manifest written to `out_dir/manifest.json`.

    Raises:
        FileNotFoundError: No index at `chroma_dir`.
        ValueError: The product collection is empty.
    """
    import chromadb

    from helioai import __version__
    from helioai.config import settings

    chroma_dir = Path(chroma_dir or settings.rag.chroma_dir)
    if not (chroma_dir / "chroma.sqlite3").exists():
        raise FileNotFoundError(f"no index at {chroma_dir}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collections: dict[str, dict] = {}
    try:
        for role, name in _collection_names().items():
            try:
                collection = client.get_collection(name)
            except Exception:
                continue
            records = out_dir / f"{role}.jsonl.gz"
            vectors: list[np.ndarray] = []
            count = 0
            with (
                records.open("wb") as raw,
                gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz,
            ):
                for offset in range(0, collection.count(), _PAGE):
                    page = collection.get(
                        include=["documents", "metadatas", "embeddings"],
                        limit=_PAGE,
                        offset=offset,
                    )
                    for pid, doc, meta in zip(
                        page["ids"], page["documents"], page["metadatas"], strict=True
                    ):
                        line = {"id": pid, "document": doc, "metadata": meta}
                        gz.write((json.dumps(line, ensure_ascii=False) + "\n").encode())
                    vectors.append(np.asarray(page["embeddings"], dtype=np.float32))
                    count += len(page["ids"])
            matrix = np.concatenate(vectors) if vectors else np.empty((0, 0), np.float32)
            embeddings = out_dir / f"{role}.npy"
            np.save(embeddings, matrix)
            collections[role] = {
                "count": count,
                "dim": int(matrix.shape[1]),
                "files": {p.name: _sha256(p) for p in (records, embeddings)},
            }
    finally:
        _release_clients()

    if not collections.get("products", {}).get("count"):
        raise ValueError(f"the product collection at {chroma_dir} is empty: nothing to export")

    try:
        from importlib.metadata import version

        speasy_version = version("speasy")
    except Exception:
        speasy_version = None
    manifest = {
        "format": FORMAT,
        "helioai": __version__,
        "speasy": speasy_version,
        "embed_model": settings.rag.embed_model,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "collections": collections,
    }
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _read_manifest(snapshot_dir: Path) -> dict:
    from helioai.config import settings

    manifest = json.loads((snapshot_dir / MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise ValueError(f"snapshot format {manifest.get('format')!r}, this HelioAI reads {FORMAT}")
    if manifest.get("embed_model") != settings.rag.embed_model:
        raise ValueError(
            f"snapshot embedded with {manifest.get('embed_model')!r}, "
            f"this install queries with {settings.rag.embed_model!r}"
        )
    if "products" not in manifest.get("collections", {}):
        raise ValueError("snapshot has no product collection")
    for entry in manifest["collections"].values():
        for fname, digest in entry["files"].items():
            if _sha256(snapshot_dir / fname) != digest:
                raise ValueError(f"{fname}: checksum mismatch, the download is incomplete")
    return manifest


def import_index(snapshot_dir: Path, chroma_dir: Path | None = None, verbose: bool = True) -> int:
    """Replace the index with a snapshot's content, and return the number of entries.

    The collections are filled in a staging directory beside the index and swapped in
    only once complete: an interrupted import leaves the previous index — or none — and
    never a partial one that `helioai index` would then take for up to date and merely
    top up. The judge's local answers (`judgment_index.jsonl`) are carried across the swap,
    as `--rebuild` does, since nothing but a paid request can recreate them.

    Raises:
        ValueError: Unknown format, another embedding model, or a checksum mismatch.
    """
    from helioai.config import settings
    from helioai.indexer import JUDGMENT_RECORDS, open_collections

    snapshot_dir = Path(snapshot_dir)
    chroma_dir = Path(chroma_dir or settings.rag.chroma_dir)
    manifest = _read_manifest(snapshot_dir)
    names = _collection_names()
    roles = [r for r in names if r in manifest["collections"]]

    staging = chroma_dir.with_name(chroma_dir.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    total = 0
    try:
        client, collections = open_collections(staging, [names[r] for r in roles])
        batch = min(client.get_max_batch_size(), _PAGE)
        for role, collection in zip(roles, collections, strict=True):
            with gzip.open(snapshot_dir / f"{role}.jsonl.gz", "rt", encoding="utf-8") as f:
                records = [json.loads(line) for line in f]
            vectors = np.load(snapshot_dir / f"{role}.npy")
            if (
                len(records) != len(vectors)
                or len(records) != manifest["collections"][role]["count"]
            ):
                raise ValueError(f"{role}: records, embeddings and manifest disagree on the count")
            for i in range(0, len(records), batch):
                chunk = records[i : i + batch]
                collection.upsert(
                    ids=[r["id"] for r in chunk],
                    documents=[r["document"] for r in chunk],
                    metadatas=[r["metadata"] or None for r in chunk],
                    embeddings=vectors[i : i + batch],
                )
                if verbose:
                    print(
                        f"[indexer]   {role}: {i + len(chunk)}/{len(records)}", end="\r", flush=True
                    )
            if verbose:
                print()
            total += len(records)
    except BaseException:
        _release_clients()
        shutil.rmtree(staging, ignore_errors=True)
        raise
    _release_clients()

    if chroma_dir.exists():
        kept = chroma_dir / JUDGMENT_RECORDS
        if kept.exists():
            shutil.copy2(kept, staging / JUDGMENT_RECORDS)
        shutil.rmtree(chroma_dir)
    staging.rename(chroma_dir)
    return total


def download_index(repo: str, version: str | None = None) -> tuple[Path, str]:
    """Fetch the snapshot published for this release, or the latest one when there is none.

    CI tags each snapshot with the release it was built by (`v0.4.0`), so an installed
    release gets the index its own code describes. A development install has no tag of
    its own and gets `main`, the most recent snapshot: an index that at worst predates
    some describing change, which `helioai index --rebuild` catches up with.
    The Hub caches the files, so a second fetch of the same revision downloads nothing.

    Returns:
        `(local_dir, revision)`.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import RevisionNotFoundError

    from helioai import __version__

    tag = f"v{version or __version__}"
    try:
        return Path(snapshot_download(repo, repo_type="dataset", revision=tag)), tag
    except RevisionNotFoundError:
        return Path(snapshot_download(repo, repo_type="dataset", revision="main")), "main"


def fetch_index(verbose: bool = True) -> int:
    """Download the published snapshot from `settings.rag.index_repo` and import it.

    Returns:
        Number of entries imported.

    Raises:
        ValueError: `HELIOAI_INDEX_REPO` is empty, or the snapshot is unusable.
        Exception: Whatever the Hub raises — network, unknown repository.
    """
    from helioai.config import settings

    repo = settings.rag.index_repo
    if not repo:
        raise ValueError("HELIOAI_INDEX_REPO is empty: fetching a prebuilt index is disabled")
    if verbose:
        print(f"[indexer] fetching the prebuilt index from huggingface.co/datasets/{repo}…")
    local, revision = download_index(repo)
    if not (local / MANIFEST).exists():
        raise ValueError(f"no snapshot published on {repo}@{revision}")
    total = import_index(local, verbose=verbose)
    manifest = json.loads((local / MANIFEST).read_text(encoding="utf-8"))
    if verbose:
        print(
            f"[indexer] {total} entries from snapshot {revision} "
            f"(HelioAI {manifest.get('helioai')}, speasy {manifest.get('speasy')}, "
            f"built {manifest.get('created')})"
        )
        print("[indexer] run `helioai index` again to add what speasy published since")
    return total


def publish_index(snapshot_dir: Path, repo: str, tag: str | None = None) -> str:
    """Upload a snapshot to `main` of the Hub dataset `repo`, and tag it with a release.

    The upload is refused when the new snapshot holds fewer than `MIN_KEPT` of the
    products the published one does. The build walks five archives over the network, and
    one of them answering 502 for the length of the walk yields an index without that
    archive and no error; published, it would replace a complete index for every new
    install. A tag already on the dataset is moved to the new commit, so re-running a
    release publishes that release's index.

    Args:
        snapshot_dir: What `export_index` wrote.
        repo: `owner/name` of the dataset; the token comes from `HF_TOKEN`.
        tag: The release, `v0.4.0` — what `download_index` asks for first.

    Returns:
        The commit id on the Hub.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RevisionNotFoundError

    snapshot_dir = Path(snapshot_dir)
    manifest = json.loads((snapshot_dir / MANIFEST).read_text(encoding="utf-8"))
    count = manifest["collections"]["products"]["count"]
    api = HfApi()
    if api.file_exists(repo, MANIFEST, repo_type="dataset"):
        published = json.loads(
            Path(api.hf_hub_download(repo, MANIFEST, repo_type="dataset")).read_text("utf-8")
        )
        previous = published["collections"]["products"]["count"]
        if count < MIN_KEPT * previous:
            raise ValueError(
                f"{count} products against {previous} published: an archive is missing, "
                "not publishing"
            )
    commit = api.upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=snapshot_dir,
        allow_patterns=[MANIFEST, "*.jsonl.gz", "*.npy"],
        commit_message=f"HelioAI {manifest['helioai']}: {count} products",
    )
    if tag:
        try:
            api.delete_tag(repo, tag=tag, repo_type="dataset")
        except RevisionNotFoundError:
            pass
        api.create_tag(repo, tag=tag, revision=commit.oid, repo_type="dataset")
    return commit.oid
