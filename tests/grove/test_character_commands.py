import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from tests.character_fixtures import timeline_snapshot, written_graph


class CharacterCommandTests(unittest.IsolatedAsyncioTestCase):
    async def test_review_project_lookup_cli_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            graph,snapshot=await written_graph(root)
            (root/'snapshot.json').write_text(json.dumps(snapshot))
            links=[lid for lid,row in snapshot['graph']['links'].items()
                   if row['left'].startswith('ep1:f') != row['right'].startswith('ep1:f')]
            (root/'changes.json').write_text(json.dumps({'reviews':[dict(link_id=lid,status='revoked',
                reviewer='fixture',reason='synthetic correction') for lid in links]}))
            graph.save(str(root/'graph.json'))
            (root/'clips.json').write_text(json.dumps([dict(id='late',source='recording',start=86405,end=86410)]))
            def run(*args):
                return subprocess.run([sys.executable,'-m','lab',*map(str,args)],text=True,capture_output=True)
            for args in [
                ['identity-review','--snapshot',root/'snapshot.json','--changes',root/'changes.json','--out',root/'state.json'],
                ['project-clips','--snapshot',root/'snapshot.json','--state',root/'state.json','--clips',root/'clips.json','--out',root/'projected.json'],
                ['grove-project','--graph',root/'graph.json','--state',root/'state.json','--out',root/'updated.json'],
                ['character-lookup','--graph',root/'updated.json','--character','person:ep1:v1','--role','speaker','--out',root/'hits.json']]:
                result=run(*args)
                self.assertEqual(result.returncode,0,result.stderr)
            hits=json.loads((root/'hits.json').read_text())['matches']
            self.assertEqual(len(hits),4)
            before=(root/'updated.json').read_bytes()
            failed=run('grove-project','--graph',root/'graph.json','--state',root/'state.json','--out',root/'updated.json')
            self.assertNotEqual(failed.returncode,0)
            self.assertEqual((root/'updated.json').read_bytes(),before)
