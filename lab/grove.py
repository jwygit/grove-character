"""Uses the patched Grove types, writer, and real graph persistence without missing upstream modules."""
import os
import copy
import sys
from pathlib import Path
from .reused.backends import request_json


def hide_identity(snapshot):
    """Unknown control: remove character mappings as well as speaker assignments."""
    snapshot = copy.deepcopy(snapshot)
    for ref in snapshot['character_refs']:
        ref['character_id'] = None
    evidence = snapshot.get('identity_evidence', {})
    for row in evidence.get('feature_mapping', {}).values():
        row['character_id'] = None
    for memory in evidence.get('memories', []):
        memory.pop('character_text', None)
    evidence.pop('writer_summary_raw', None)
    # These are not writer input; omit them in the control to avoid accidental leakage.
    for key in ('graph', 'raw', 'equivalences', 'feature_registry'):
        snapshot.pop(key, None)
    return snapshot


def load_grove():
    path = Path(__file__).resolve().parents[1]/'vendor'/'Grove'
    sys.path.insert(0, str(path))
    return path


def make_llm(config):
    async def llm(prompt, **kwargs):
        import asyncio
        headers = {}
        key = os.environ.get(config.get('api_key_env', 'LLM_API_KEY'))
        if key:
            from urllib.parse import urlparse
            u = urlparse(config['endpoint'])
            if u.scheme != 'https' and u.hostname not in ('localhost', '127.0.0.1'):
                raise ValueError('use HTTPS for authenticated remote endpoints')
            headers['Authorization'] = 'Bearer '+key
        raw = await asyncio.to_thread(request_json, config['endpoint'], {
            'model': config['model'], 'temperature': 0,
            'messages': [{'role': 'user', 'content': prompt}],
            'response_format': {'type': 'json_object'}}, headers, config.get('timeout', 300))
        return raw['choices'][0]['message']['content']
    return llm


async def write_episode(episode_data, snapshot, llm, output):
    load_grove()
    from hypervideo.types import VideoEpisode
    from hypervideo.structure import VideoHypergraph
    from hypervideo.modules.character_identity import (SnapshotProvider, identity_context,
        CharacterFactExtractor, resolve_episode_identity, rewrite_summary)
    from types import SimpleNamespace
    import json
    episode = VideoEpisode.from_dict(dict(episode_data))
    provider = SnapshotProvider([snapshot], snapshot['source'])
    episode.character_refs, episode.identity_evidence = await resolve_episode_identity(
        provider, episode.start_seconds, episode.end_seconds)
    prompt = ('Summarize this fixed episode. Return JSON {"summary":"...", "subject":"..."}. '
              'The boundary is fixed; use speaker identity only where supported.\n' +
              episode.get_frame_texts() + identity_context(episode.character_refs, episode.identity_evidence))
    summary = rewrite_summary(json.loads(await llm(prompt)), episode.identity_evidence)
    episode.summary, episode.subject = summary['summary'], summary['subject']
    result = await CharacterFactExtractor(llm).extract_facts(SimpleNamespace(topic_id=''), [episode])
    graph = VideoHypergraph()
    graph.add_episode_node(episode)
    for fact in result.facts:
        graph.add_fact_node(fact)
    # This isolated writer intentionally does not simulate topic aggregation/retrieval.
    graph.save(str(output))
    restored = VideoHypergraph.load(str(output))
    if graph.to_dict() != restored.to_dict():
        raise AssertionError('Grove graph round-trip changed data')
    return graph.to_dict()
