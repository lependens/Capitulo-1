"""Copy documented DEVELOPMENT examples from the versioned scientific memoir."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
memo = root / 'scientific_pipeline/docs/ET0_V2_MEMORIA_CIENTIFICA.md'
source = memo.read_text(encoding='utf-8')

def rows(start, end, prefix):
    section = source.split(start, 1)[1].split(end, 1)[0]
    return [[cell.strip() for cell in line.strip('|').split('|')]
            for line in section.splitlines() if line.startswith(prefix)]

data = {
    'ahc': rows('### 7.2.', '## 8.', '| ')[2:19],
    'ml': rows('### 8.2.', '## 9.', '| ')[2:9],
    'ann': rows('### 9.2.', '## 10.', '| ANN_')[:10],
    'annFrequency': rows('### 9.2.', '## 10.', '| ANN_')[10:20],
}
assert [len(value) for value in data.values()] == [17, 7, 10, 10]
assert all(len(row) == 6 for name in ('ahc', 'ml', 'ann') for row in data[name])
target = root / 'web_app/frontend/src/et0-v2-memo.json'
target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
