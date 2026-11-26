import copy
from pathlib import Path
import unittest
from lab.character_links import (bind, build_index, face_occurrences, lookup,
                                 project_clip, seal_state, validate_state)
from lab.identity_state import export_state
from tests.character_fixtures import timeline_snapshot, revoke


class CharacterLinksTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = timeline_snapshot()
        self.state = self.snapshot['identity_state']
        self.events = self.snapshot['temporal_evidence']

    def project(self, start=86405, end=86410, source='recording', state=None):
        return project_clip(dict(source=source,start=start,end=end),self.events,state or self.state)

    def test_all_samples_survive_gallery_limit_and_noise_remains_unknown(self):
        refs = self.project()
        self.assertEqual([r['start'] for r in refs if r['role']=='visible'],[86407,86408,86409])
        self.assertIsNone(next(r for r in refs if r['start']==86407)['character_id'])
        self.assertEqual(len([e for e in self.events if e['role']=='visible']),7)
        self.assertTrue(all(r['start']==r['end'] for r in refs if r['role']=='visible'))

    def test_half_open_windows_voice_intersection_and_source_isolation(self):
        self.assertNotIn(86404,[r['start'] for r in self.project(86400,86404) if r['role']=='visible'])
        refs = self.project(86402,86404)
        self.assertEqual([(r['start'],r['end']) for r in refs if r['role']=='speaker'],[(86402,86403)])
        self.assertEqual(self.project(source='other camera'),[])

    def test_short_turn_retained_as_unknown_speaker(self):
        ref = next(r for r in self.project() if r.get('utterance_id')=='u2')
        self.assertIsNone(ref['character_id'])
        self.assertEqual(ref['identity_observation_ids'],[])

    def test_revocation_reassigns_voice_without_changing_face_samples(self):
        state = revoke(self.snapshot)['identity_state']
        refs = self.project(state=state)
        self.assertEqual(next(r for r in refs if r.get('utterance_id')=='u3')['character_id'],'person:ep1:v1')
        self.assertEqual(next(r for r in refs if r['start']==86409)['character_id'],'person:ep1:f1')
        self.assertEqual(self.snapshot['identity_state'],self.state)

    def test_split_group_is_ambiguous_instead_of_arbitrarily_choosing_person(self):
        ids = self.snapshot['identity_evidence']['feature_mapping']['<voice_0>']['identity_observation_ids']
        state = revoke(self.snapshot,split_voice=True)['identity_state']
        row = bind(ids,state)
        self.assertIsNone(row['character_id'])
        self.assertEqual(row['binding_status'],'ambiguous')
        self.assertEqual(len(row['candidate_character_ids']),2)

    def test_state_is_content_addressed_not_only_revision(self):
        other = copy.deepcopy(self.state); other['names']={'Alice':['ep1:f1']}
        with self.assertRaises(ValueError): validate_state(other)
        other = seal_state(other)
        self.assertEqual(other['revision'],self.state['revision'])
        self.assertNotEqual(other['state_id'],self.state['state_id'])

    def test_reverse_index_resolves_alias_names_roles_and_typed_ids(self):
        state = export_state(self.snapshot['graph'], {'Alice':['ep1:f1']})
        refs = self.project(state=state)
        records = [dict(id='same',node_type=kind,character_refs=refs) for kind in ['fact','visual_clip']]
        index = build_index(records,state)
        self.assertEqual(len(lookup(index,state,['Alice','person:ep1:v1'])),2)
        self.assertEqual(lookup(index,state,['Alice'],['referenced']),[])
        with self.assertRaises(ValueError): lookup(index,state,['unregistered name'])

    def test_old_references_and_missing_observations_fail_closed(self):
        records=[dict(id='x',node_type='visual_clip',character_refs=self.project())]
        with self.assertRaisesRegex(ValueError,'stale'): build_index(records,revoke(self.snapshot)['identity_state'])
        state=copy.deepcopy(self.state); del state['observations']['ep1:f1']; state=seal_state(state)
        with self.assertRaises(ValueError): self.project(state=state)

    def test_temporal_evidence_rejects_source_modality_and_duplicate_errors(self):
        for change in ['source','modality','duplicate']:
            with self.subTest(change=change):
                events=copy.deepcopy(self.events)
                if change=='source': events[0]['source']='wrong'
                elif change=='modality': events[0]['identity_observation_ids']=['ep1:v1']
                else: events.append(events[0])
                with self.assertRaises(ValueError):
                    project_clip(dict(source='recording',start=86400,end=86410),events,self.state)

    def test_grove_and_independent_contract_are_identical(self):
        root=Path(__file__).resolve().parents[2]
        self.assertEqual((root/'lab/character_links.py').read_bytes(),
            (root/'vendor/Grove/hypervideo/modules/character_links.py').read_bytes())
