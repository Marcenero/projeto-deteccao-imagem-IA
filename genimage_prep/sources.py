"""Leitura de imagens oficiais extraídas ou do espelho Arrow fixado."""

from collections.abc import Iterator
from dataclasses import dataclass
import io
from pathlib import Path, PurePosixPath
from random import Random
import re


GENERATOR = "stable_diffusion_v_1_4"
OFFICIAL_URL = "https://github.com/GenImage-Dataset/GenImage"
HUB_ID = "nebula/GenImage-arrow"
REVISION = "3f4b9f921a673be09a93b335ed728cea0c6ecf33"
HUB_URL = f"https://huggingface.co/datasets/{HUB_ID}"
CONFIGURATIONS = {"train": "sd14-train", "test": "sd14-test"}
SHARDS = {"train": 189, "test": 8}
ARCHIVE_BYTES = {"train": 93_059_841_288, "test": 3_558_541_152}
FOLDERS = {0: "nature", 1: "ai"}
SUFFIXES = {".jpeg", ".jpg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}


@dataclass(frozen=True)
class Candidate:
    image_id: str
    label: int
    source: Path | bytes | None

    def read(self) -> bytes:
        if isinstance(self.source, Path):
            return self.source.read_bytes()
        if self.source is None:
            raise ValueError("Campo image sem bytes")
        return self.source

    @property
    def category(self) -> str:
        path = PurePosixPath(self.image_id)
        if wnid := re.search(r"(?:^|[/_])(n[0-9]{8})(?=[/_.]|$)", str(path)):
            return wnid[1]
        if self.label == 1 and (index := re.match(r"([0-9]{3})_sdv4_", path.name)):
            return f"sd14_index:{index[1]}"
        return ""


class LocalSource:
    def __init__(self, root: Path, version: str | None):
        self.root = root.resolve()
        valid_names = {"stablediffusionv14", "stablediffusion14"}
        if re.sub(r"[^a-z0-9]", "", root.resolve().name.lower()) not in valid_names:
            raise ValueError("Aponte --local-root para a pasta oficial Stable Diffusion V1.4.")
        for split in ("train", "val"):
            for folder in FOLDERS.values():
                if not (self.root / split / folder).is_dir():
                    raise ValueError(f"Pasta oficial ausente: {self.root / split / folder}")
        self.provenance = {
            "mode": "local",
            "url": OFFICIAL_URL,
            "root": str(self.root),
            "version": version,
            "official_partitions": {"train": "train", "test": "val"},
        }
        self.transfer_bytes = None

    def groups(self, split: str, seed: int) -> Iterator[tuple[str, Iterator[Candidate]]]:
        partition = self.provenance["official_partitions"][split]
        for label, folder in FOLDERS.items():
            base = self.root / partition / folder
            paths = sorted(
                path.relative_to(self.root).as_posix()
                for path in base.rglob("*")
                if path.suffix.lower() in SUFFIXES and path.is_file()
            )
            Random(f"{seed}:{split}:{label}").shuffle(paths)
            candidates = (Candidate(name, label, self.root / name) for name in paths)
            yield f"{partition}/{folder}", candidates


class ArrowStream(io.RawIOBase):
    """Conta os bytes da resposta parcial entregues ao leitor Arrow."""

    def __init__(self, response, source):
        super().__init__()
        self.response = response
        self.source = source

    def readable(self):
        return True

    def readinto(self, target):
        remaining = self.source.budget - self.source.transfer_bytes
        if not remaining:
            return 0
        try:
            count = self.response.raw.readinto(memoryview(target)[:remaining])
        except Exception as exc:
            raise OSError("A conexão foi interrompida durante a leitura do Arrow.") from exc
        self.source.transfer_bytes += count
        return count


class HubSource:
    def __init__(self, budget_mib: int, max_batches: int):
        self.budget = budget_mib * 1024**2
        self.max_batches = max_batches
        self.transfer_bytes = 0
        self.provenance = {
            "mode": "streaming",
            "url": HUB_URL,
            "revision": REVISION,
            "version": REVISION,
            "configurations": CONFIGURATIONS,
            "official_partitions": {"train": "train", "test": "val"},
            "arrow_disk_bytes": ARCHIVE_BYTES,
            "stream_budget_bytes": self.budget,
            "max_batches_per_shard": self.max_batches,
        }

    def groups(self, split: str, seed: int) -> Iterator[tuple[str, Iterator[Candidate]]]:
        count = SHARDS[split]
        shards = list(range(count))
        Random(f"{seed}:{split}:shards").shuffle(shards)
        for number in shards:
            name = f"data-{number:05d}-of-{count:05d}.arrow"
            shard = f"data/{split}/{GENERATOR}/{name}"
            yield shard, self._candidates(shard, split, seed)

    def _candidates(self, shard: str, split: str, seed: int) -> Iterator[Candidate]:
        from itertools import islice

        import pyarrow as pa
        import requests

        remaining = self.budget - self.transfer_bytes
        if remaining <= 0:
            raise RuntimeError("Limite de transferência do streaming atingido.")
        url = f"{HUB_URL}/resolve/{REVISION}/{shard}"
        headers = {"Range": f"bytes=0-{remaining - 1}", "Accept-Encoding": "identity"}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(15, 40)) as response:
                response.raise_for_status()
                content_range = re.fullmatch(
                    r"bytes 0-([0-9]+)/[0-9]+", response.headers.get("Content-Range", "")
                )
                if (
                    response.status_code != 206
                    or not content_range
                    or int(content_range[1]) >= remaining
                    or response.headers.get("Content-Encoding", "identity") != "identity"
                ):
                    raise RuntimeError("O servidor não aceitou uma resposta parcial dentro do limite.")
                with io.BufferedReader(ArrowStream(response, self)) as stream:
                    with pa.ipc.open_stream(stream) as reader:
                        for index, batch in enumerate(islice(reader, self.max_batches)):
                            order = list(range(batch.num_rows))
                            Random(f"{seed}:{shard}:{index}").shuffle(order)
                            paths = batch.column("image_path")
                            labels = batch.column("label")
                            images = batch.column("image")
                            generators = batch.column("generator")
                            for row in order:
                                name, label = paths[row].as_py(), labels[row].as_py()
                                expected = FOLDERS.get(label)
                                parts = PurePosixPath(name).parts
                                if (
                                    generators[row].as_py() != 4
                                    or expected is None
                                    or parts[:3] != (GENERATOR, split, expected)
                                ):
                                    raise RuntimeError("Rótulo, gerador ou partição Arrow inesperado.")
                                yield Candidate(name, label, images[row].as_py())
        except (requests.RequestException, OSError, pa.ArrowException) as exc:
            raise RuntimeError(f"Falha no streaming de {shard}: {exc}") from exc
