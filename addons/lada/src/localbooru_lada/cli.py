import argparse
import json
import socket
import sys
from pathlib import Path

from .constants import LADA_REVISION, PROTOCOL_VERSION
from .probe import ProbeConfig, probe_runtime
from .release import audit_base_artifact, build_common_runtime, build_release_manifest, build_runtime_layer
from .server import ServerConfig, SidecarServer


def _probe(args) -> int:
    with Path(args.config).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    config = ProbeConfig(
        protocol_version=value.get("protocol_version", PROTOCOL_VERSION),
        upstream_revision=value.get("upstream_revision", LADA_REVISION),
        expected_upstream_revision=LADA_REVISION,
        models=value["models"],
        model_revision=value.get("model_revision", ""),
        requested_backend=args.backend or value.get("requested_backend", "auto"),
        fp16=value.get("fp16", True),
        model_probe_size=value.get("model_probe_size", 64),
        model_probe_frames=value.get("model_probe_frames", 2),
        max_probe_seconds=value.get("max_probe_seconds", 90.0),
    )
    result = probe_runtime(config)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ready"] else 2


def _serve(args) -> int:
    config = ServerConfig.load(Path(args.config))
    connection = socket.socket(fileno=args.socket_fd)
    SidecarServer(connection, config).run()
    return 0


def _manifest(args) -> int:
    root = Path(args.root)
    bundles = {}
    for entry in args.bundle:
        name, path = entry.split("=", 1)
        bundles[name] = Path(path)
    installed_sizes = {}
    for entry in args.installed_size:
        name, size = entry.split("=", 1)
        installed_sizes[name] = int(size)
    manifest = build_release_manifest(
        root,
        bundles,
        source_archive=Path(args.source_archive),
        installed_sizes=installed_sizes,
        cuda_variant=args.cuda_variant,
    )
    output = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)
    return 0


def _build_common(args) -> int:
    build_common_runtime(Path(args.cuda), Path(args.xpu), Path(args.output))
    return 0


def _build_layer(args) -> int:
    build_runtime_layer(Path(args.base), Path(args.complete), Path(args.output))
    return 0


def _audit_base(args) -> int:
    if args.inventory == "-":
        entries = [line.rstrip("\n") for line in sys.stdin]
    else:
        entries = Path(args.inventory).read_text(encoding="utf-8").splitlines()
    audit_base_artifact(entries)
    print(json.dumps({"ok": True, "entries": len(entries)}, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="localbooru-lada-sidecar")
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe = subparsers.add_parser("probe", help="verify models and an accelerated backend")
    probe.add_argument("--config", required=True)
    probe.add_argument("--backend", choices=("auto", "cuda", "xpu"))
    probe.set_defaults(run=_probe)

    serve = subparsers.add_parser("serve", help="serve one inherited LocalBooru session")
    serve.add_argument("--config", required=True)
    serve.add_argument("--socket-fd", type=int, required=True)
    serve.set_defaults(run=_serve)

    manifest = subparsers.add_parser("build-manifest", help="hash release and source bundles")
    manifest.add_argument("--root", required=True)
    manifest.add_argument("--source-archive", required=True)
    manifest.add_argument("--bundle", action="append", default=[])
    manifest.add_argument("--installed-size", action="append", default=[])
    manifest.add_argument("--cuda-variant", choices=("cuda", "cuda-legacy"), default="cuda")
    manifest.add_argument("--output")
    manifest.set_defaults(run=_manifest)

    common = subparsers.add_parser("build-common", help="derive common files from CUDA and XPU runtimes")
    common.add_argument("--cuda", required=True)
    common.add_argument("--xpu", required=True)
    common.add_argument("--output", required=True)
    common.set_defaults(run=_build_common)

    layer = subparsers.add_parser("build-layer", help="create a runtime delta from a complete tree")
    layer.add_argument("--base", required=True)
    layer.add_argument("--complete", required=True)
    layer.add_argument("--output", required=True)
    layer.set_defaults(run=_build_layer)

    audit = subparsers.add_parser("audit-base", help="reject LADA payloads in a base release inventory")
    audit.add_argument("--inventory", required=True)
    audit.set_defaults(run=_audit_base)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
