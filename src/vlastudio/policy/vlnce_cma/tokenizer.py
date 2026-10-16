"""CMA's original Habitat vocabulary and tokenizer, owned by the policy."""
import json
from pathlib import Path
import re


class InstructionTokenizer:
    def __init__(self, vocabulary=None, download=True):
        self.vocabulary = vocabulary
        self.download = download
        self.lookup = None

    def _load(self):
        if self.vocabulary is None:
            self.vocabulary = json.loads(Path(__file__).with_name('instruction_vocab.json').read_text())
        if self.lookup is None:
            self.lookup = {word: index for index, word in enumerate(self.vocabulary)}

    def __call__(self, text, length=200):
        self._load()
        text = text.lower()
        # Match the released annotation preprocessing; retain hyphenated words.
        for token in ("'s",):
            text = text.replace(token, ' '+token)
        text = text.replace(',', '').replace('?', '')
        tokens = [token.strip() for token in re.split(r'([^\w-]+)', text) if token.strip()]
        ids = [self.lookup.get(token, self.lookup['<unk>']) for token in tokens[:length]]
        return ids+[self.lookup['<pad>']]*(length-len(ids))

    def save(self, directory):
        self._load()
        (Path(directory)/'instruction_vocab.json').write_text(json.dumps(self.vocabulary)+'\n')
