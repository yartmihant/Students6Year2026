"""Check saved results, local HTML assets and animation frames; make a QA contact sheet."""
import base64
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import re

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ('href', 'src') and value:
                self.links.append(value)


summary = []
for path in sorted(ROOT.glob('*.ipynb')):
    nb = json.loads(path.read_text(encoding='utf-8'))
    cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    assert all(c['execution_count'] is not None for c in cells), path
    outputs = [o for c in cells for o in c['outputs']]
    assert not any(o['output_type'] == 'error' for o in outputs), path
    assert not any(o.get('name') == 'stderr' for o in outputs), path
    images = sum('image/png' in o.get('data', {}) for o in outputs)
    movies = sum('text/html' in o.get('data', {}) for o in outputs)
    assert images == 3 and movies == (2 if path.name.startswith('1.2') else 3)
    assert 'Все численные проверки пройдены' in outputs[-1]['text']
    summary.append(f'{path.name}: {len(cells)} executed cells, {images} figures, {movies} animations')

for path in list(ROOT.glob('*.html')) + list((ROOT/'animations').glob('*.html')):
    parser = Links()
    parser.feed(path.read_text(encoding='utf-8'))
    for link in parser.links:
        assert not link.startswith(('http:', 'https:')), (path, link)
        if not link.startswith(('data:', '#')):
            assert (path.parent/link).exists(), (path, link)

movies = sorted((ROOT/'animations').glob('*.html'))
assert len(movies) == 8
sheet = Image.new('RGB', (1200, 8*260), 'white')
draw = ImageDraw.Draw(sheet)
for row, path in enumerate(movies):
    source = path.read_text(encoding='utf-8')
    frames = re.findall(r'frames\[\d+\] = "(data:image/png;base64,.*?)"', source, re.S)
    assert len(frames) >= 90, path
    decoded = []
    for frame in frames:
        data = frame.split(',', 1)[1].replace('\\\n', '').replace('\\\r\n', '')
        image = Image.open(io.BytesIO(base64.b64decode(data)))
        image.load()
        decoded.append(image)
    assert decoded[0].tobytes() != decoded[len(decoded)//2].tobytes(), path
    for col, j in enumerate([0, len(decoded)//2, len(decoded)-1]):
        thumb = decoded[j].convert('RGB')
        thumb.thumbnail((395, 232))
        sheet.paste(thumb, (col*400+(400-thumb.width)//2, row*260+22))
        draw.text((col*400+5, row*260+3), f'{path.stem} / frame {j}', fill='black')
    summary.append(f'{path.name}: {len(frames)} decodable embedded frames, no external dependencies')
sheet.save(ROOT/'previews'/'animation_contact_sheet.jpg', quality=92)
(ROOT/'verification.txt').write_text('\n'.join(summary)+'\n', encoding='utf-8')
print('\n'.join(summary))
