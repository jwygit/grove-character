"""Shared validation and exact interval-duration metrics. Standard library only."""
import math


def validate_segments(segments, duration, face_ids):
    if not isinstance(segments, list):
        raise ValueError("segments must be a list")
    for s in segments:
        if not isinstance(s, dict) or not {"start", "end", "face_id"} <= s.keys():
            raise ValueError("segment requires start/end/face_id")
        a, b = s["start"], s["end"]
        if not all(type(v) in (int, float) and math.isfinite(v) for v in (a, b)):
            raise ValueError("timestamps must be finite numbers")
        if not 0 <= a < b <= duration:
            raise ValueError("segment outside clip or empty")
        if s["face_id"] is not None and s["face_id"] not in face_ids:
            raise ValueError(f"unknown face ID: {s['face_id']}")
    return segments


def validate_manifest(data):
    if data.get("schema_version") != 1 or not data.get("clips"):
        raise ValueError("manifest requires schema_version=1 and nonempty clips")
    seen = set()
    for clip in data["clips"]:
        if not isinstance(clip.get("id"), str) or not clip["id"] or clip["id"] in seen:
            raise ValueError("clip IDs must be unique nonempty strings")
        seen.add(clip["id"])
        d = clip["duration"]
        if type(d) not in (float, int) or not math.isfinite(d) or d <= 0:
            raise ValueError("duration must be positive")
        ids = [f["id"] for f in clip["faces"]]
        if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("invalid face registry")
        for f in clip["faces"]:
            if not f.get("image"):
                raise ValueError("face reference image required")
        if "utterances" in clip:
            utterances = clip["utterances"]
            uids = [u["id"] for u in utterances]
            if len(set(uids)) != len(uids):
                raise ValueError("duplicate utterance IDs")
            validate_segments([dict(u, face_id=None) for u in utterances], d, ids)
    return data


def normalize_prediction(payload, clip, mode):
    ids = {f["id"] for f in clip["faces"]}
    segments = payload["segments"]
    if mode == "association":
        utterances = {u["id"]: u for u in clip["utterances"]}
        found = [s.get("utterance_id") for s in segments]
        if len(found) != len(set(found)) or set(found) != set(utterances):
            raise ValueError("association output must cover each utterance exactly once")
        segments = [dict(s, start=utterances[s["utterance_id"]]["start"],
                         end=utterances[s["utterance_id"]]["end"]) for s in segments]
    return validate_segments(segments, clip["duration"], ids)


def score_clip(truth, prediction, duration, face_ids):
    """Exact piecewise integration, includes overlap, no collar or label remapping.

    Face IDs reference a common face gallery, NOT arbitrary diarization labels.
    null means audible speech without an assigned visible face.
    """
    validate_segments(truth, duration, face_ids)
    validate_segments(prediction, duration, face_ids)
    points = sorted({0, duration, *(s[k] for s in truth + prediction for k in ("start", "end"))})
    counts = dict(tp=0., fp=0., fn=0., speech_tp=0., speech_fp=0., speech_fn=0.,
                  offscreen_seconds=0., offscreen_false_link_seconds=0.)
    for a, b in zip(points, points[1:]):
        t, dt = (a + b) / 2, b - a
        g = [s for s in truth if s["start"] <= t < s["end"]]
        p = [s for s in prediction if s["start"] <= t < s["end"]]
        gs, ps = {s["face_id"] for s in g} - {None}, {s["face_id"] for s in p} - {None}
        counts["tp"] += dt * len(gs & ps)
        counts["fp"] += dt * len(ps - gs)
        counts["fn"] += dt * len(gs - ps)
        counts["speech_tp"] += dt * bool(g and p)
        counts["speech_fp"] += dt * bool(p and not g)
        counts["speech_fn"] += dt * bool(g and not p)
        if g and not gs:  # exclusively off-screen speech; mixed overlap excluded
            counts["offscreen_seconds"] += dt
            counts["offscreen_false_link_seconds"] += dt * bool(ps)
    return counts


def metrics(counts):
    def ratio(a, b):
        return a / b if b else None
    tp, fp, fn = (counts[k] for k in ("tp", "fp", "fn"))
    return {"association_precision": ratio(tp, tp + fp),
            "association_recall": ratio(tp, tp + fn),
            "association_f1": ratio(2 * tp, 2 * tp + fp + fn),
            "offscreen_false_link_rate": ratio(counts["offscreen_false_link_seconds"], counts["offscreen_seconds"]),
            "speech_activity_f1": ratio(2 * counts["speech_tp"],
                2 * counts["speech_tp"] + counts["speech_fp"] + counts["speech_fn"])}


def fuse_specialists(clip, payload, mode, threshold, margin):
    """Join diarization turns to time-local active-speaker scores.

    ASD score is model-specific, not a probability. Competing speakers are left
    unresolved. No automatic permanent identity merge is performed.
    """
    turns = clip["utterances"] if mode == "association" else payload["turns"]
    faces = {f["id"] for f in clip["faces"]}
    validate_segments([dict(t, face_id=None) for t in turns], clip["duration"], faces)
    windows = payload["active_speaker_scores"]
    validate_segments(windows, clip["duration"], faces)
    for w in windows:
        if w["face_id"] is None or type(w.get("score")) not in (float, int) or not math.isfinite(w["score"]):
            raise ValueError("ASD scores require a face ID and finite score")
    for face in faces:
        spans = sorted((w["start"], w["end"]) for w in windows if w["face_id"] == face)
        if any(b > c for (a, b), (c, d) in zip(spans, spans[1:])):
            raise ValueError("ASD intervals for one face must not overlap")
    result = []
    for i, turn in enumerate(turns):
        overlap = any(j != i and min(turn["end"], other["end"]) > max(turn["start"], other["start"])
                      for j, other in enumerate(turns))
        ranked = []
        for face in faces:
            total, weight = 0., 0.
            for w in windows:
                if w["face_id"] != face:
                    continue
                dt = max(0, min(turn["end"], w["end"]) - max(turn["start"], w["start"]))
                total += dt * w["score"]
                weight += dt
            # Require coverage of >= half the utterance, rather than one good frame.
            if weight >= .5 * (turn["end"] - turn["start"]):
                ranked.append((total / weight, face))
        ranked.sort(reverse=True)
        face_id = None
        if not overlap and ranked and ranked[0][0] >= threshold:
            if len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= margin:
                face_id = ranked[0][1]
        item = {"start": turn["start"], "end": turn["end"], "face_id": face_id,
                "evidence": {"ranked_asd_scores": ranked, "overlap_abstention": overlap}}
        if mode == "association":
            item["utterance_id"] = turn["id"]
        result.append(item)
    return {"segments": result}
