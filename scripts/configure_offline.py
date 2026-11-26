"""Generate staged deployment configs from downloaded revisions and local weights."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / "configs" / name).read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def configure(stage, models, visual_url, vlm_url, embedding_url, speech_input, accept_links):
    manifest = json.loads((models / "manifest.json").read_text())

    def revision(name):
        value = manifest[name]["revision"]
        if not value or "SET_" in value or not (models / name).is_dir():
            raise ValueError(f"missing local model/revision: {name}")
        return value

    config = read("full_loop.json")
    # Bring-up preset. A shared episode cap is used in all three staged runs.
    # This is a deliberate experiment parameter, not M3/Grove's original setting.
    config["grove"].update(use_visual_channel=stage != "core", max_buffer_seconds=30)
    for key in ("caption", "writer"):
        config[key].update(endpoint=vlm_url, model="grove-vlm",
                           model_revision=revision("Qwen2.5-VL-7B-Instruct"))
    config["embedding"].update(endpoint=embedding_url, model="grove-embedding",
                               model_revision=revision("Qwen3-Embedding-0.6B"))
    if stage != "core":
        with urlopen(visual_url.rstrip("/") + "/health", timeout=15) as response:
            health = json.load(response)
        expected = revision("siglip-base-patch16-224")
        if health["backbone"] != "siglip" or health["model_revision"] != expected:
            raise ValueError("visual server does not match the downloaded model manifest")
        config["visual_encoder"].update(server_url=visual_url, backbone=health["backbone"],
                                       dim=health["dim"], model_revision=expected)
    if stage == "characters":
        if not speech_input or not speech_input.is_file():
            raise ValueError("characters require --speech-input from the speech command")
        speech = json.loads(speech_input.read_text())
        if speech.get("status") != "completed" or not speech.get("audio_sha256"):
            raise ValueError("speech input must be completed and retain PCM provenance")
        face_root = models / "insightface"
        onnx = sorted((face_root / "models" / "buffalo_l").glob("*.onnx"))
        required = {"1k3d68.onnx", "2d106det.onnx", "det_10g.onnx", "genderage.onnx", "w600k_r50.onnx"}
        if not required.issubset({p.name for p in onnx}):
            raise ValueError("download/extract the complete buffalo_l model pack first")
        fingerprint = hashlib.sha256(json.dumps([(p.name, sha(p)) for p in onnx]).encode()).hexdigest()
        perception = read("perception.json")
        voice = models / "pretrained_eres2netv2.ckpt"
        perception.update(insightface_root=str(face_root.resolve()), face_model_id=fingerprint,
                          voice_checkpoint=str(voice.resolve()), voice_model_id=sha(voice))
        association = read("av_omni.json")
        association.update(model_path=str((models / "M3-Agent-Memorization").resolve()),
                           model_revision=revision("M3-Agent-Memorization"), attention="sdpa")
        grounding = read("visual_grounding.json")
        grounding.update(endpoint=vlm_url, model="grove-vlm", max_frames=40,
                         model_revision=config["caption"]["model_revision"])
        config["characters"] = dict(speech_input=str(speech_input.resolve()),
            perception=perception, association=association, grounding=grounding, accept_links=accept_links)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["core", "visual", "characters"], required=True)
    parser.add_argument("--models", type=Path, default=Path("/models"))
    parser.add_argument("--visual-url", default="http://127.0.0.1:9090")
    parser.add_argument("--vlm-url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--embedding-url", default="http://127.0.0.1:8001/v1/embeddings")
    parser.add_argument("--speech-input", type=Path)
    parser.add_argument("--accept-model-links", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new config name to preserve experiment settings")
    config = configure(args.stage, args.models.resolve(), args.visual_url, args.vlm_url,
                       args.embedding_url, args.speech_input, args.accept_model_links)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
