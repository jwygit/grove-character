"""M3 joint memorization, adapted to episode evidence and Grove identity snapshots.

Prompt and input ordering follow pinned M3 sources (Apache-2.0, see provenance).
No active-speaker timestamps are inferred from an identity equivalence statement.
"""
import copy
import hashlib
import inspect
import json
import re
from pathlib import Path
from .reused.m3_prompt import M3_PROMPT
from .reused.backends import parse_json

FEATURE = re.compile(r'<(?:face|voice)_[^<>]+>')
EQUIVALENCE = re.compile(r'Equivalence:\s*(<face_\d+>)\s*,\s*(<voice_\d+>)\s*')


def prepare_features(prepared):
    graph = prepared.graph
    labels = graph.data.setdefault('m3_feature_labels', {'face': {}, 'voice': {}})
    registry = {}
    def label(modality, node):
        table = labels[modality]
        if node not in table:
            table[node] = f'<{modality}_{len(table)}>'
        return table[node]
    for face in prepared.clip['faces']:
        anchor = graph.data['face_groups'][face['observation_id']]['anchor_id']
        tag = label('face', anchor)
        row = registry.setdefault(tag, dict(modality='face', node_id=anchor,
            anchor_id=anchor, face_ids=[], observation_ids=[], image=face['image']))
        row['face_ids'].append(face['id'])
        row['observation_ids'].extend(face.get('observation_ids', [face['observation_id']]))
    for group in prepared.clip['voice_groups']:
        vid = group['voice_id']
        registry[label('voice', vid)] = dict(modality='voice', node_id=vid,
            anchor_id=graph.data['voice_nodes'][vid]['anchor_id'],
            observation_ids=[u['voice_observation_id'] for u in prepared.bundle['utterances']
                             if prepared.voice_ids.get(u['id']) == vid],
            utterances=copy.deepcopy(group['utterances']))
    prepared.clip['m3_features'] = registry
    return registry


def timestamp(seconds):
    # MM:SS input structure from M3, retaining sub-second frontend precision.
    minutes, milliseconds = divmod(round(seconds*1000), 60000)
    return f'{minutes:02d}:{milliseconds/1000:06.3f}'


def build_messages(clip, config):
    video = {'type': 'video', 'video': clip['video']}
    for key in ('fps', 'max_pixels'):
        if key in config:
            video[key] = config[key]
    content = [{'type': 'text', 'text': M3_PROMPT}, video,
               {'type': 'text', 'text': 'Face features:'}]
    voices = {}
    for tag, feature in clip['m3_features'].items():
        if feature['modality'] == 'face':
            content.extend([{'type': 'text', 'text': tag+':'},
                            {'type': 'image', 'image': feature['image']}])
        else:
            voices[tag] = [dict(start_time=timestamp(u['start']), end_time=timestamp(u['end']),
                                asr=u['text']) for u in feature['utterances']]
    content.extend([{'type': 'text', 'text': 'Voice features:'},
                    {'type': 'text', 'text': json.dumps(voices, ensure_ascii=False)}])
    return [{'role': 'user', 'content': content}]


def parse_memories(payload, registry):
    """Handle the upstream singular-prompt/plural-reader mismatch explicitly."""
    keys = [k for k in ('video_description', 'video_descriptions') if k in payload]
    if not keys or ('high_level_conclusions' not in payload):
        raise ValueError('M3 output requires video_description and high_level_conclusions')
    if len(keys) == 2 and payload[keys[0]] != payload[keys[1]]:
        raise ValueError('conflicting singular/plural M3 descriptions')
    memories, pairs, linked = [], [], {}
    for kind, rows in [('description', payload[keys[0]]),
                       ('conclusion', payload['high_level_conclusions'])]:
        if not isinstance(rows, list) or any(not isinstance(s, str) for s in rows):
            raise ValueError('M3 memories must be arrays of strings')
        for index, text in enumerate(rows):
            if '<person:' in text:
                raise ValueError('M3 must reference supplied features, not character IDs')
            tags = list(dict.fromkeys(FEATURE.findall(text)))
            if any(tag not in registry for tag in tags):
                raise ValueError('M3 output references an unknown feature ID')
            if 'Equivalence:' in text:
                match = EQUIVALENCE.fullmatch(text.strip())
                if kind != 'conclusion' or match is None:
                    raise ValueError('malformed M3 equivalence statement')
                face, voice = match.groups()
                if voice in linked and linked[voice] != face:
                    raise ValueError('conflicting faces for one voice in M3 output')
                linked[voice] = face
                pair = {'face': face, 'voice': voice}
                if pair not in pairs:
                    pairs.append(pair)
                continue
            memories.append(dict(id=f'{kind}:{index}', kind=kind, text=text, feature_ids=tags))
    return memories, pairs, keys


def finish_memorization(prepared, raw, accept_links=False):
    registry = prepared.clip['m3_features']
    memories, pairs, keys = parse_memories(raw, registry)
    graph, bundle = prepared.graph, prepared.bundle
    for pair in pairs:
        face, voice = registry[pair['face']], registry[pair['voice']]
        link_id = f"m3_eq:{bundle['id']}:{pair['face']}:{pair['voice']}"
        if link_id not in graph.data['links']:
            graph.propose(dict(id=link_id, left=face['anchor_id'], right=voice['anchor_id'],
                method='m3_equivalence', evidence=dict(feature_ids=pair, source=bundle['source'],
                    statement=f"Equivalence: {pair['face']}, {pair['voice']}",
                    policy_accept=accept_links)))
        if accept_links and graph.data['links'][link_id]['status'] == 'candidate':
            graph.review(link_id, 'confirmed', 'experiment-policy', 'accept M3 equivalence')
    exported = graph.export()
    people = {e['observation_id']: e['person_id'] for e in exported['events']}
    mapping = {tag: dict(feature_id=row['node_id'], modality=row['modality'],
                        character_id=people[row['anchor_id']],
                        evidence_ids=list(dict.fromkeys([row['anchor_id']]+row['observation_ids'])),
                        graph_revision=exported['revision']) for tag, row in registry.items()}
    voice_tags = {row['node_id']: tag for tag, row in registry.items() if row['modality'] == 'voice'}
    refs = []
    for u in bundle['utterances']:
        vid = prepared.voice_ids.get(u['id'])
        tag = voice_tags.get(vid)
        refs.append(dict(utterance_id=u['id'], start=bundle['time_origin']+u['start'],
            end=bundle['time_origin']+u['end'], text=u['text'], voice_id=vid, feature_id=tag,
            character_id=mapping[tag]['character_id'] if tag else None, role='speaker',
            evidence_ids=[u['voice_observation_id']] if tag else [],
            visible_face_id=None, visibility='not_predicted', graph_revision=exported['revision']))
    registry_ids = {tag: dict(node_id=row['node_id'], observation_ids=row['observation_ids'])
                    for tag, row in registry.items()}
    registry_hash = hashlib.sha256(json.dumps(registry_ids, sort_keys=True).encode()).hexdigest()
    return dict(source=bundle['source'], start=bundle['time_origin'],
        end=bundle['time_origin']+bundle['duration'], graph_revision=exported['revision'],
        character_refs=refs, graph=copy.deepcopy(graph.data), association_schema='m3_equivalence_v1',
        identity_evidence=dict(schema_version=1, graph_revision=exported['revision'],
            feature_mapping=mapping, memories=memories),
        feature_registry=copy.deepcopy(registry), feature_registry_sha256=registry_hash, equivalences=pairs,
        equivalences_accepted=accept_links, description_keys=keys,
        voice_groups=copy.deepcopy(prepared.clip['voice_groups']),
        excluded_voice_turns=prepared.exclusions, raw=raw)


class M3Backend:
    mode = 'm3_memorization'

    def __init__(self, config, root):
        self.config, self.root = config, Path(root)
        self.model = self.processor = None

    def infer(self, clip, mode):
        if mode != self.mode:
            raise ValueError('M3 checkpoint requires the joint memorization task')
        path = (self.root/self.config['model_path']).resolve()
        if not path.is_dir():
            raise ValueError('local M3 checkpoint missing; downloads are not automatic')
        revision = self.config.get('model_revision', '')
        if not revision or revision.startswith('SET_'):
            raise ValueError('record the downloaded M3 model revision or weight hash in config')
        import torch
        from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration, GenerationConfig
        from qwen_omni_utils import process_mm_info
        if not torch.cuda.is_available():
            raise RuntimeError('M3 memorization requires a CUDA host')
        if self.model is None:
            self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                str(path), torch_dtype='auto', device_map='auto', local_files_only=True,
                attn_implementation=self.config.get('attention', 'flash_attention_2'))
            self.model.eval()
            self.processor = Qwen2_5OmniProcessor.from_pretrained(str(path), local_files_only=True)
        messages = build_messages(clip, self.config)
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
        # Upstream M3 uses `audios`; newer Transformers renamed it to `audio`.
        # Never pass an ignored kwarg that would silently drop the soundtrack.
        parameters = inspect.signature(self.processor.__call__).parameters
        audio_key = next((k for k in ('audios', 'audio') if k in parameters), None)
        if audio_key is None:
            raise RuntimeError('unsupported Omni processor audio interface')
        inputs = self.processor(text=text, **{audio_key: audios}, images=images, videos=videos,
            return_tensors='pt', padding=True, use_audio_in_video=True)
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        devices = sorted({p.device.index for p in self.model.parameters() if p.device.type == 'cuda'})
        for device in devices:
            torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            generation = self.model.generate(**inputs,
                generation_config=GenerationConfig(pad_token_id=151643, bos_token_id=151644, eos_token_id=151645),
                use_audio_in_video=True, max_new_tokens=self.config.get('max_new_tokens', 4096),
                do_sample=True, temperature=self.config.get('temperature', 1e-6))
        generated = generation[:, inputs.input_ids.size(1):]
        response = self.processor.batch_decode(generated, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)[0]
        result = parse_json(response)
        result['raw_response'] = response
        result['inference'] = dict(model_id=self.config['model_id'], model_revision=revision,
            prompt_sha256=hashlib.sha256(M3_PROMPT.encode()).hexdigest(), audio_parameter=audio_key,
            peak_allocated_bytes={str(d): torch.cuda.max_memory_allocated(d) for d in devices})
        return result
