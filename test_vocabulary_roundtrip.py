"""Full-record regression checks using the shipped prompt and vocabularies."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import apply_custom_vocabulary as forward
import restore_canonical_vocabulary as reverse


class VocabularyRoundTripTests(unittest.TestCase):
    def test_full_record_roundtrip(self):
        prompt = (ROOT / 'assets/prompt').read_text(encoding='utf-8')
        real = prompt.replace(r'\n', '\n')
        lines = real.split('\n')
        separators = ['\n', r'\n', '\r\n', '\r']
        mixed = ''.join(line + separators[i % len(separators)]
                        for i, line in enumerate(lines[:-1])) + lines[-1]
        prompts = {
            'literal': prompt,
            'real': real,
            'crlf': real.replace('\n', '\r\n'),
            'mixed': mixed,
            'no_terminal_newline': real.rstrip('\r\n'),
        }
        mappings = sorted(p for p in (ROOT / 'alternative_ciphers').glob('*.json')
                           if p.name != 'manifest.json')
        with tempfile.TemporaryDirectory() as directory:
            source, encoded, restored = [Path(directory) / name for name in
                                         ('source.jsonl', 'encoded.jsonl', 'restored.jsonl')]
            for mapping in mappings:
                for style, prompt_variant in prompts.items():
                    with self.subTest(mapping=mapping.name, newlines=style):
                        record = {
                            'id': 'roundtrip-1',
                            'prompt-base': prompt_variant,
                            'query': 'wadapapawacacapawapanapa',
                            'answer': 'ga-ma-ma-za-za',
                            'image': 'aW1hZ2UtYnl0ZXM=',
                            'metadata': {'untouched': [r'\n', 'wa:ca', 42]},
                        }
                        source.write_text(json.dumps(record) + '\n', encoding='utf-8')
                        with contextlib.redirect_stdout(io.StringIO()):
                            forward.convert_file(source, encoded, mapping, True)
                        remapped = json.loads(encoded.read_text(encoding='utf-8'))
                        self.assertNotEqual(remapped['prompt-base'], record['prompt-base'])
                        reverse.convert_file(encoded, restored, mapping, True)
                        self.assertEqual(json.loads(restored.read_text(encoding='utf-8')), record)


if __name__ == '__main__':
    unittest.main()
