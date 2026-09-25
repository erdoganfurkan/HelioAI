"""helioai.index_snapshot: export → import round trip, refusals, and where it downloads from."""

from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

from helioai import index_snapshot
from helioai.config import settings
from helioai.indexer import JUDGMENT_RECORDS, open_collections


def _vectors(rng, n):
    v = rng.normal(size=(n, 8)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _build(chroma_dir, products: int = 7, catalogs: int = 2):
    """Normalised, as the indexer stores them: Chroma renormalises a cosine collection's
    vectors on insert, so a round trip moves them by at most an ulp."""
    rng = np.random.default_rng(0)
    _, (p, c) = open_collections(
        chroma_dir, [settings.rag.collection_name, settings.rag.catalogs_collection_name]
    )
    if products:
        p.upsert(
            ids=[f"cda/P{i}" for i in range(products)],
            documents=[f"product {i}. Région: é." for i in range(products)],
            metadatas=[{"provider": "cda", "n": i, "x": 0.5, "ok": True} for i in range(products)],
            embeddings=_vectors(rng, products),
        )
    if catalogs:
        c.upsert(
            ids=[f"amda/c{i}" for i in range(catalogs)],
            documents=[f"catalog {i}" for i in range(catalogs)],
            metadatas=[{"product_type": "catalog"}] * catalogs,
            embeddings=_vectors(rng, catalogs),
        )
    index_snapshot._release_clients()


def _content(chroma_dir, name):
    import chromadb

    got = (
        chromadb.PersistentClient(path=str(chroma_dir))
        .get_collection(name)
        .get(include=["documents", "metadatas", "embeddings"])
    )
    index_snapshot._release_clients()
    order = np.argsort(got["ids"])
    return (
        [got["ids"][i] for i in order],
        [got["documents"][i] for i in order],
        [got["metadatas"][i] for i in order],
        np.asarray(got["embeddings"])[order],
    )


def test_a_snapshot_round_trips_and_replaces_the_index_but_not_the_judges_answers(tmp_path):
    source = tmp_path / "source"
    _build(source)
    manifest = index_snapshot.export_index(tmp_path / "snap", source)

    assert manifest["format"] == index_snapshot.FORMAT
    assert manifest["embed_model"] == settings.rag.embed_model
    assert {r: c["count"] for r, c in manifest["collections"].items()} == {
        "products": 7,
        "catalogs": 2,
    }
    assert json.loads((tmp_path / "snap" / "manifest.json").read_text()) == manifest

    target = settings.rag.chroma_dir
    _build(target, products=3, catalogs=0)
    (target / JUDGMENT_RECORDS).write_text('{"id": "paid"}\n')
    assert index_snapshot.import_index(tmp_path / "snap", verbose=False) == 9

    for name in (settings.rag.collection_name, settings.rag.catalogs_collection_name):
        want, got = _content(source, name), _content(target, name)
        assert want[:3] == got[:3]
        np.testing.assert_allclose(want[3], got[3], rtol=0, atol=1e-7)
    assert (target / JUDGMENT_RECORDS).read_text() == '{"id": "paid"}\n'
    assert not target.with_name(target.name + ".partial").exists()
    _, (products,) = open_collections(target, [settings.rag.collection_name])
    assert products.configuration_json["hnsw"]["sync_threshold"] == 1


@pytest.mark.parametrize(
    "spoil, message",
    [
        (lambda d, m: m.update(embed_model="another/model"), "embedded with"),
        (lambda d, m: m.update(format=99), "format"),
        (lambda d, m: (d / "products.npy").write_bytes(b"truncated"), "checksum"),
        (lambda d, m: m.pop("collections"), "no product collection"),
    ],
)
def test_an_unusable_snapshot_is_refused_and_the_index_left_alone(tmp_path, spoil, message):
    _build(tmp_path / "source")
    snap = tmp_path / "snap"
    manifest = index_snapshot.export_index(snap, tmp_path / "source")
    spoil(snap, manifest)
    (snap / "manifest.json").write_text(json.dumps(manifest))
    _build(settings.rag.chroma_dir, products=3, catalogs=0)

    with pytest.raises(ValueError, match=message):
        index_snapshot.import_index(snap, verbose=False)
    assert _content(settings.rag.chroma_dir, settings.rag.collection_name)[0] == [
        "cda/P0",
        "cda/P1",
        "cda/P2",
    ]


def test_an_interrupted_import_leaves_no_staging_and_the_previous_index(tmp_path, monkeypatch):
    _build(tmp_path / "source")
    index_snapshot.export_index(tmp_path / "snap", tmp_path / "source")
    _build(settings.rag.chroma_dir, products=3, catalogs=0)
    monkeypatch.setattr(index_snapshot.np, "load", lambda p: np.zeros((1, 8), np.float32))

    with pytest.raises(ValueError, match="disagree"):
        index_snapshot.import_index(tmp_path / "snap", verbose=False)
    target = settings.rag.chroma_dir
    assert not target.with_name(target.name + ".partial").exists()
    assert len(_content(target, settings.rag.collection_name)[0]) == 3


def test_export_refuses_an_absent_or_empty_index(tmp_path):
    with pytest.raises(FileNotFoundError):
        index_snapshot.export_index(tmp_path / "snap", tmp_path / "absent")
    _build(tmp_path / "empty", products=0, catalogs=0)
    with pytest.raises(ValueError, match="empty"):
        index_snapshot.export_index(tmp_path / "snap", tmp_path / "empty")


def test_index_is_empty(tmp_path):
    assert index_snapshot.index_is_empty()
    _build(settings.rag.chroma_dir, products=0, catalogs=1)
    assert index_snapshot.index_is_empty()
    _build(settings.rag.chroma_dir, products=1, catalogs=0)
    assert not index_snapshot.index_is_empty()


def test_a_release_without_its_own_snapshot_gets_the_latest(tmp_path, monkeypatch):
    import huggingface_hub
    from huggingface_hub.errors import RevisionNotFoundError

    asked = []

    def fake(repo, *, repo_type, revision):
        asked.append((repo, repo_type, revision))
        if revision != "main":
            raise RevisionNotFoundError(
                "no tag",
                response=httpx.Response(404, request=httpx.Request("GET", "https://hf.co")),
            )
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    assert index_snapshot.download_index("org/idx", "9.9.9") == (tmp_path, "main")
    assert asked == [("org/idx", "dataset", "v9.9.9"), ("org/idx", "dataset", "main")]


def test_fetch_imports_the_downloaded_snapshot(tmp_path, monkeypatch, capsys):
    _build(tmp_path / "source")
    index_snapshot.export_index(tmp_path / "snap", tmp_path / "source")
    monkeypatch.setattr(
        index_snapshot, "download_index", lambda repo: (tmp_path / "snap", "v0.4.0")
    )

    assert index_snapshot.fetch_index() == 9
    assert "9 entries from snapshot v0.4.0" in capsys.readouterr().out
    assert not index_snapshot.index_is_empty()


def test_fetch_is_disabled_by_an_empty_repository_setting(monkeypatch):
    monkeypatch.setattr(settings.rag, "index_repo", "")
    with pytest.raises(ValueError, match="HELIOAI_INDEX_REPO"):
        index_snapshot.fetch_index()


def test_fetch_says_when_nothing_is_published_yet(tmp_path, monkeypatch):
    monkeypatch.setattr(index_snapshot, "download_index", lambda repo: (tmp_path, "main"))
    with pytest.raises(ValueError, match="no snapshot published"):
        index_snapshot.fetch_index(verbose=False)


class _FakeHub:
    """The four calls `publish_index` makes, against an in-memory dataset."""

    def __init__(self, tmp_path, published: int | None, tags=()):
        self.tmp_path, self.published, self.tags = tmp_path, published, set(tags)
        self.uploads: list[dict] = []

    def __call__(self):
        return self

    def file_exists(self, repo, filename, *, repo_type):
        return self.published is not None

    def hf_hub_download(self, repo, filename, *, repo_type):
        path = self.tmp_path / "published.json"
        path.write_text(json.dumps({"collections": {"products": {"count": self.published}}}))
        return str(path)

    def upload_folder(self, **kw):
        from types import SimpleNamespace

        self.uploads.append(kw)
        return SimpleNamespace(oid=f"commit{len(self.uploads)}")

    def delete_tag(self, repo, *, tag, repo_type):
        from huggingface_hub.errors import RevisionNotFoundError

        if tag not in self.tags:
            raise RevisionNotFoundError(
                "absent",
                response=httpx.Response(404, request=httpx.Request("GET", "https://hf.co")),
            )
        self.tags.discard(tag)

    def create_tag(self, repo, *, tag, revision, repo_type):
        self.tags.add((tag, revision))


@pytest.fixture
def snapshot(tmp_path):
    _build(tmp_path / "source")
    index_snapshot.export_index(tmp_path / "snap", tmp_path / "source")
    return tmp_path / "snap"


@pytest.mark.parametrize("published, tags", [(None, ()), (7, ()), (7, ("v1.0.0",))])
def test_publish_uploads_and_tags_the_release(snapshot, tmp_path, monkeypatch, published, tags):
    import huggingface_hub

    hub = _FakeHub(tmp_path, published, tags)
    monkeypatch.setattr(huggingface_hub, "HfApi", hub)

    assert index_snapshot.publish_index(snapshot, "org/idx", tag="v1.0.0") == "commit1"
    (upload,) = hub.uploads
    assert upload["repo_id"] == "org/idx" and upload["repo_type"] == "dataset"
    assert "7 products" in upload["commit_message"]
    assert hub.tags == {("v1.0.0", "commit1")}, "a re-run moves the release's tag"


def test_publish_refuses_a_snapshot_that_lost_an_archive(snapshot, tmp_path, monkeypatch):
    import huggingface_hub

    hub = _FakeHub(tmp_path, published=100)
    monkeypatch.setattr(huggingface_hub, "HfApi", hub)

    with pytest.raises(ValueError, match="7 products against 100"):
        index_snapshot.publish_index(snapshot, "org/idx", tag="v1.0.0")
    assert hub.uploads == [] and hub.tags == set()
