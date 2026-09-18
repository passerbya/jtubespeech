"""Portable local pyannote 3.x bundles and a network guard for inference."""
import copy
import os
import re
import shutil
import socket
import tempfile
from contextlib import contextmanager
from pathlib import Path
from videoid_state import locked

COMPONENTS = ("segmentation", "embedding")


@contextmanager
def pyannote_checkpoint_globals(torch_module):
    """Allow known pyannote 3.x metadata while keeping weights-only loading enabled."""
    serialization = getattr(torch_module, "serialization", None)
    safe_globals = getattr(serialization, "safe_globals", None)
    if safe_globals is None:
        # Earlier torch versions have no public safe-globals API.
        yield
        return
    from torch.torch_version import TorchVersion
    from pyannote.audio.core.task import Problem, Resolution, Specifications

    approved = (TorchVersion, Specifications, Problem, Resolution)
    existing = getattr(serialization, "get_safe_globals", lambda: [])()
    # safe_globals removes its entries on exit, so don't re-add existing user entries.
    added = [item for item in approved if item not in existing]
    with safe_globals(added):
        yield


def set_offline_environment():
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK"):
        os.environ[key] = "1"
    os.environ["PYANNOTE_METRICS_ENABLED"] = "0"


@contextmanager
def network_disabled():
    """Block Python internet sockets/DNS, including connections through local proxies."""
    set_offline_environment()
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    dns_names = ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr")
    original_dns = {name: getattr(socket, name) for name in dns_names}

    def blocked_dns(*args, **kwargs):
        raise RuntimeError("Offline inference blocked a DNS lookup; prepare the local model bundle first")

    def connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError("Offline inference blocked a network connection")
        return original_connect(sock, address)

    def connect_ex(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError("Offline inference blocked a network connection")
        return original_connect_ex(sock, address)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    for name in dns_names:
        setattr(socket, name, blocked_dns)
    try:
        yield
    finally:
        socket.socket.connect = original_connect
        socket.socket.connect_ex = original_connect_ex
        for name, function in original_dns.items():
            setattr(socket, name, function)


def read_config(path):
    import yaml
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if (not isinstance(config, dict) or not isinstance(config.get("pipeline"), dict)
            or config["pipeline"].get("name") != "pyannote.audio.pipelines.SpeakerDiarization"):
        raise ValueError("Expected a local pyannote 3.x SpeakerDiarization pipeline config")
    if config.get("preprocessors"):
        raise ValueError("Custom pipeline preprocessors are not supported by the offline loader")
    params = config["pipeline"].get("params")
    if not isinstance(params, dict):
        raise ValueError("Pipeline config has no params")
    for component in COMPONENTS:
        value = params.get(component)
        reference = value.get("checkpoint") if isinstance(value, dict) else value
        if not isinstance(reference, str) or not reference:
            raise ValueError(f"Pipeline config has no valid {component} checkpoint")
    return config


def reference_for(value):
    return value["checkpoint"] if isinstance(value, dict) else value


def replaced_reference(value, path):
    if isinstance(value, dict):
        result = dict(value)
        result.pop("use_auth_token", None)
        result.pop("token", None)
        result["checkpoint"] = str(path)
        return result
    return str(path)


def download_error_details(error, token_env):
    """Preserve nested transport errors without exposing tokens or signed URLs."""
    details = []
    seen = set()
    secret = os.environ.get(token_env)
    while error is not None and id(error) not in seen and len(details) < 8:
        seen.add(id(error))
        message = str(error)
        if secret:
            message = message.replace(secret, "<redacted>")
        message = re.sub(r"hf_[A-Za-z0-9]+", "<redacted>", message)
        message = re.sub(r"(?i)(Bearer\s+)\S+", r"\1<redacted>", message)
        message = re.sub(r"(https?://)[^/\s]+@", r"\1<redacted>@", message)
        message = re.sub(r"(https?://[^\s?]+)\?[^\s]+", r"\1?<redacted>", message)
        details.append(f"{type(error).__name__}: {message[:1200]}")
        error = error.__cause__ or error.__context__
    return "\n  caused by ".join(details)


def hub_file(model, filename, cache_dir, token_env, local_only):
    from huggingface_hub import hf_hub_download
    model_id, separator, revision = model.partition("@")
    if not local_only:
        print(f"[DOWNLOAD] repo={model_id} file={filename} "
              f"{token_env}={'set' if os.environ.get(token_env) else 'not set (cached login may be used)'}",
              flush=True)
    try:
        return Path(hf_hub_download(
            repo_id=model_id, filename=filename, revision=revision if separator else None,
            cache_dir=str(cache_dir), local_files_only=local_only,
            token=False if local_only else (os.environ.get(token_env) or None),
        ))
    except Exception as exc:
        mode = "Local cache lookup" if local_only else "Model download"
        raise RuntimeError(f"{mode} failed: repo={model_id} file={filename}\n  "
                           + download_error_details(exc, token_env)) from exc


def bundle_directory(cache_root, model):
    name = re.sub(r"[^A-Za-z0-9_.-]+", "--", str(model)).strip(".-")
    if not name:
        raise ValueError("Invalid model identifier")
    return Path(cache_root) / "local" / name


def write_yaml_atomic(path, config):
    import yaml
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", dir=str(path.parent),
                                         encoding="utf-8", delete=False) as stream:
            temporary = Path(stream.name)
            yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def prepare_model_bundle(model, cache_root, token_env="HF_TOKEN", local_only=False):
    """Export config plus both weights. local_only migrates an existing old cache."""
    cache_root = Path(cache_root).expanduser().resolve()
    hub_cache = cache_root / "pyannote"
    source = Path(model).expanduser()
    if source.is_dir():
        source = source / "config.yaml"
    if not source.is_file():
        if source.is_absolute() or source.suffix.lower() in {".yaml", ".yml"}:
            raise FileNotFoundError(f"Local pipeline not found: {source}")
        source = hub_file(str(model), "config.yaml", hub_cache, token_env, local_only)
    config = read_config(source)
    params = config["pipeline"]["params"]
    resolved = {}
    for component in COMPONENTS:
        reference = reference_for(params[component])
        weight = Path(reference).expanduser()
        if not weight.is_absolute():
            weight = source.parent / weight
        if not weight.is_file():
            if (Path(reference).is_absolute() or reference.startswith((".", "~"))
                    or Path(reference).suffix.lower() in {".bin", ".pt", ".ckpt", ".onnx"}):
                raise FileNotFoundError(f"Local {component} weight not found: {weight}")
            weight = hub_file(reference, "pytorch_model.bin", hub_cache, token_env, local_only)
        if weight.stat().st_size == 0:
            raise ValueError(f"Empty {component} weight: {weight}")
        resolved[component] = weight.resolve()

    destination = bundle_directory(cache_root, model)
    destination.mkdir(parents=True, exist_ok=True)
    config_path = destination / "config.yaml"
    with (destination / ".bundle.lock").open("a+b", buffering=0) as lock, locked(lock):
        if local_only and config_path.is_file():
            return config_path
        config = copy.deepcopy(config)
        params = config["pipeline"]["params"]
        params.pop("use_auth_token", None)
        params.pop("token", None)
        config.pop("device", None)
        for component, weight in resolved.items():
            # The 'pyannote' prefix prevents Wespeaker checkpoints being mistaken for ONNX.
            target = destination / f"pyannote_{component}.bin"
            if weight != target.resolve():
                with tempfile.NamedTemporaryFile(dir=str(destination), suffix=".bin.tmp",
                                                 delete=False) as stream:
                    temporary = Path(stream.name)
                try:
                    shutil.copyfile(weight, temporary)
                    os.replace(temporary, target)
                finally:
                    if temporary.exists():
                        temporary.unlink()
            params[component] = replaced_reference(params[component], target.name)
        # Publishing config last makes it the marker for a complete bundle.
        write_yaml_atomic(config_path, config)
    return config_path


def offline_config_path(model, token_env="HF_TOKEN"):
    source = Path(model).expanduser()
    if source.is_dir():
        source = source / "config.yaml"
    if source.is_file():
        return source.resolve()
    if source.is_absolute() or source.suffix.lower() in {".yaml", ".yml"}:
        raise FileNotFoundError(f"Local pipeline not found: {source}")
    cache = os.environ.get("PYANNOTE_CACHE")
    if not cache:
        raise RuntimeError("Offline inference requires --model /local/config.yaml or --cache-dir "
                           "containing a downloaded model")
    root = Path(cache).parent
    prepared = bundle_directory(root, model) / "config.yaml"
    if prepared.is_file():
        return prepared
    try:
        # No HTTP/version check: Hugging Face is only asked to find cached files.
        return prepare_model_bundle(model, root, token_env, local_only=True)
    except Exception as exc:
        raise RuntimeError("Local model bundle/cache is incomplete. Run --download-model on an "
                           "internet-connected machine and copy its local/ directory to this server. "
                           f"No download was attempted. Details: {exc}") from exc


@contextmanager
def local_pipeline_config(model, token_env="HF_TOKEN"):
    """Resolve relative weights next to the config, independent of the working directory."""
    path = offline_config_path(model, token_env)
    config = read_config(path)
    params = config["pipeline"]["params"]
    params.pop("use_auth_token", None)
    params.pop("token", None)
    config.pop("device", None)
    for component in COMPONENTS:
        reference = reference_for(params[component])
        weight = Path(reference).expanduser()
        if not weight.is_absolute():
            weight = path.parent / weight
        weight = weight.resolve()
        if not weight.is_file() or weight.stat().st_size == 0:
            raise FileNotFoundError(f"Missing local {component} weight: {weight}; no online fallback")
        params[component] = replaced_reference(params[component], weight)
    with tempfile.TemporaryDirectory(prefix="pyannote-local-") as directory:
        runtime_config = Path(directory) / "config.yaml"
        write_yaml_atomic(runtime_config, config)
        yield runtime_config
