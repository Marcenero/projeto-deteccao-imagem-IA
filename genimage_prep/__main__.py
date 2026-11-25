"""Executa a preparação da amostra GenImage SD 1.4."""

import argparse
from pathlib import Path
import shutil

import yaml

from .prepare import SPLITS, prepare
from .sources import HubSource, LocalSource


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--profile", choices=("smoke", "experiment"), default="smoke")
    parser.add_argument("--source", choices=("streaming", "local"), default="streaming")
    parser.add_argument("--local-root", type=Path)
    parser.add_argument("--local-version", help="Versão ou SHA-256 do arquivo oficial extraído")
    args = parser.parse_args()
    if args.source == "local" and args.local_root is None:
        parser.error("--source local requer --local-root.")
    if args.source == "streaming" and (args.local_root or args.local_version):
        parser.error("--local-root e --local-version requerem --source local.")

    try:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        profile = config["profiles"][args.profile]
        quotas = profile["per_class"]
        if set(quotas) != set(SPLITS) or any(type(n) is not int or n < 1 for n in quotas.values()):
            raise ValueError("Informe cotas inteiras positivas para train, validation e test.")
        seed = config["seed"]
        if type(seed) is not int:
            raise ValueError("A seed deve ser inteira.")
        size = tuple(config["image_size"])
        if size != (256, 256):
            raise ValueError("A resolução de análise desta etapa é 256 × 256.")
        output = Path(config["output_dir"]) / args.profile
        summary_path = Path(config["summary_path"])
        if args.source == "local":
            source = LocalSource(args.local_root, args.local_version)
        else:
            budget = profile["stream_budget_mib"]
            batches = profile["max_batches_per_shard"]
            if any(type(n) is not int or n < 1 for n in (budget, batches)):
                raise ValueError("Os limites de streaming devem ser inteiros positivos.")
            output.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(output).free
            if free < budget * 1024**2:
                raise ValueError(f"Espaço livre de {free} bytes abaixo do teto de {budget} MiB.")
            source = HubSource(budget, batches)
            print(f"Streaming SD 1.4, teto de {budget} MiB e {free} bytes livres.", flush=True)
        summary = prepare(source, profile=args.profile, quotas=quotas, seed=seed, size=size,
                          output=output, summary_path=summary_path)
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        parser.exit(1, f"Erro de preparação: {exc}\n")

    if summary["status"] != "complete":
        parser.exit(1, f"{summary['error']} Consulte {summary_path}.\n")
    print(f"{summary['selected']['total']} imagens em {summary['manifest']}. "
          f"Relatório em {summary_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
