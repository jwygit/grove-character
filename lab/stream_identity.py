"""Run existing perception/M3/grounding only after Grove closes a dynamic episode."""
import asyncio
import copy
from pathlib import Path
from uuid import uuid4
from .association import associate
from .common import audio, digest as file_digest, read, validate_turns, duration
from .fact_evidence import digest
from .perception import extract
from .visual_grounding import run_grounding
from .observability import emit, stage


def identity_provider_factory(video, output, config, source, origin, *,
                              perception_extract=extract, association_backend=None, observer=None):
    """GPU model boundaries are injectable for contract tests; defaults run real models.

ASR runs once on shared PCM, independently of visual episode decisions. Completed
episode bounds select/clamp turns without moving the Grove boundary. State is
derived only from committed episodes; abandoned inference artifacts cannot become
the previous person graph.
"""
    from .m3 import M3Backend
    output = Path(output)
    audio_path, speech = prepare_speech(video, output, config, observer)
    backend = association_backend or M3Backend(config["association"], output)

    def factory(graph):
        prior_paths = [e.identity_evidence.get("provider_snapshot_path") for e in
                       sorted(graph.episodes.values(), key=lambda e: e.end_seconds)]
        prior_paths = [path for path in prior_paths if path]
        prior = read(prior_paths[-1])["snapshot"]["graph"] if prior_paths else None
        return EpisodeIdentityProvider(video, output, config, source, origin, speech,
                                        audio_path, prior, backend, perception_extract, observer)
    return factory


def prepare_speech(video, output, config, observer=None):
    from hypervideo.streaming_memory import atomic_json
    output = Path(output)
    audio_path = output / "identity-source.wav"
    speech_path = output / "identity-speech.json"
    if not audio_path.exists():
        temp = output / ("audio-" + uuid4().hex + ".wav")
        audio(video, temp)
        temp.replace(audio_path)
    audio_hash = file_digest(audio_path)
    speech_contract = dict(video_sha256=file_digest(video), audio_sha256=audio_hash,
                           speech_config=config.get("speech"), speech_input=config.get("speech_input"))
    if speech_path.exists():
        saved = read(speech_path)
        if saved["contract"] != speech_contract:
            raise ValueError("cached ASR input/config mismatch")
        speech = saved["speech"]
    else:
        if config.get("speech_input"):
            speech = read(config["speech_input"])
            if speech.get("audio_sha256") != audio_hash:
                raise ValueError("precomputed speech must match the shared PCM SHA256")
        else:
            from .speech import infer
            with stage(observer, 'speech', mode='offline_full_audio'):
                speech = infer(audio_path, config["speech"])
            speech["audio_sha256"] = audio_hash
        validate_turns(speech["segments"], duration(audio_path))
        atomic_json(speech_path, dict(contract=speech_contract, speech=speech))
    emit(observer, 'speech', 'artifact', files=[str(speech_path), str(audio_path)],
         count=len(speech['segments']), mode='offline_full_audio')
    return audio_path, speech


class EpisodeIdentityProvider:
    def __init__(self, video, output, config, source, origin, speech, audio_path,
                 prior, backend, perception_extract, observer=None):
        self.video, self.output, self.config = video, output, config
        self.source, self.origin, self.speech = source, origin, speech
        self.audio_path, self.prior = audio_path, copy.deepcopy(prior)
        self.backend, self.extract = backend, perception_extract
        self.observer = observer

    async def episode_evidence(self, start, end):
        from hypervideo.modules.character_identity import SnapshotProvider
        from hypervideo.streaming_memory import atomic_json
        if not self.origin <= start < end:
            raise ValueError("episode precedes source time origin")
        contract = dict(source=self.source, start=start, end=end, config=self.config,
                        video_sha256=file_digest(self.video), prior_sha256=digest(self.prior),
                        speech_sha256=digest(self.speech))
        key = digest(contract)
        path = self.output / ("identity-" + key + ".json")
        if path.exists():
            saved = read(path)
            if saved["contract"] != contract:
                raise ValueError("identity cache contract mismatch")
            snapshot = saved["snapshot"]
        else:
            directory = self.output / ("identity-work-" + uuid4().hex)
            directory.mkdir()
            with stage(self.observer, 'face_voice', episode_start=start-self.origin, episode_end=end-self.origin):
                bundle = await asyncio.to_thread(self.extract, self.video, self.speech,
                    start-self.origin, end-self.origin, self.origin, self.source,
                    "episode:" + key[:24], self.config["perception"], directory / "perception",
                    source_wav=self.audio_path)
            bundle['config'] = copy.deepcopy(self.config['perception'])
            atomic_json(directory / 'bundle.json', bundle)
            emit(self.observer, 'face_voice', 'artifact', files=[str(directory/'bundle.json')],
                 face_count=len(bundle['faces']), utterance_count=len(bundle['utterances']))
            with stage(self.observer, 'association', accept_links=self.config.get('accept_links', False)):
                snapshot = await asyncio.to_thread(associate, bundle, self.backend,
                    prior=self.prior, accept_links=self.config.get("accept_links", False))
            atomic_json(directory/'association.json', snapshot)
            emit(self.observer, 'association', 'artifact', files=[str(directory/'association.json')])
            ground = directory / "grounding"
            ground.mkdir()
            with stage(self.observer, 'grounding'):
                snapshot = await asyncio.to_thread(run_grounding, bundle, snapshot, self.config["grounding"], ground)
            atomic_json(directory / "bundle.json", bundle)
            atomic_json(path, dict(contract=contract, snapshot=snapshot))
        emit(self.observer, 'identity', 'artifact', files=[str(path)],
             episode_start=start-self.origin, episode_end=end-self.origin)
        snapshot = copy.deepcopy(snapshot)
        snapshot["identity_evidence"]["provider_snapshot_path"] = str(path)
        # Same validation and evidence contract as the independent module.
        return await SnapshotProvider([snapshot], self.source).episode_evidence(start, end)
