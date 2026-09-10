"""Execute notebooks with a real Jupyter kernel; export their saved outputs to HTML.

Usage: python tools/execute_notebooks.py [notebook.ipynb ...]
No nbconvert/nbclient required. All code cells must explicitly display figures.
"""
import base64
import html
import json
import os
from pathlib import Path
import queue
import re
import sys
import time

from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLBACKEND', 'Agg')


def render_markdown(text):
    try:
        import mistune
        return mistune.html(text)
    except ImportError:
        def inline(s):
            # Preserve LaTeX verbatim; underscore is part of mathematical notation.
            s = html.escape(html.unescape(s))
            s = re.sub(r'`([^`]+)`', r'<code>\1</code>', s)
            return re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', s)
        result, paragraph, in_list = [], [], False
        def flush():
            if paragraph:
                result.append('<p>'+inline(' '.join(paragraph))+'</p>')
                paragraph.clear()
        lines = text.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith('|'):
                flush()
                rows = []
                while i < len(lines) and lines[i].startswith('|'):
                    row = lines[i].strip('|').split('|')
                    if not all(re.fullmatch(r'\s*:?-+:?\s*', item) for item in row):
                        tag = 'th' if not rows else 'td'
                        rows.append('<tr>'+''.join(f'<{tag}>'+inline(item.strip())+f'</{tag}>' for item in row)+'</tr>')
                    i += 1
                result.append('<table>'+''.join(rows)+'</table>')
                continue
            heading = re.match(r'^(#{1,6})\s+(.*)', line)
            if heading:
                flush()
                if in_list: result.append('</ul>'); in_list = False
                n = len(heading[1]); result.append(f'<h{n}>'+inline(heading[2])+f'</h{n}>')
            elif line.startswith('- '):
                flush()
                if not in_list: result.append('<ul>'); in_list = True
                result.append('<li>'+inline(line[2:])+'</li>')
            elif not line.strip():
                flush()
                if in_list: result.append('</ul>'); in_list = False
            else:
                paragraph.append(line.strip())
            i += 1
        flush()
        if in_list: result.append('</ul>')
        return '\n'.join(result)


def export(path, nb):
    parts = ['<!doctype html><html lang="ru"><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width, initial-scale=1">',
             '<title>' + html.escape(path.stem) + '</title>',
             '<style>body{max-width:1180px;margin:32px auto;padding:0 20px;font:16px/1.55 system-ui;color:#182432}'
             'pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5f7;padding:16px;font:13px/1.45 monospace}'
             'img{max-width:100%;height:auto}table{border-collapse:collapse;font-size:14px}'
             'td,th{padding:6px 12px;border:1px solid #ccd4dc}h1,h2{line-height:1.25}'
             '.output{margin:16px 0}details{margin:18px 0}summary{cursor:pointer;color:#225e91}'
             'iframe{width:100%;height:570px;border:1px solid #ddd}.markdown{font:inherit}</style>',
             '<p><a href="index.html">← Все задания</a></p>',
             '<p>Расчёты уже выполнены. Код раскрывается нажатием. Анимации работают без сети. '
             'Математические формулы в этой HTML-копии сохранены в записи LaTeX; '
             'для их набора откройте исходный notebook в Jupyter.</p>']
    previews = ROOT/'previews'
    previews.mkdir(exist_ok=True)
    image_number, animation_number = 0, 0
    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown':
            parts.append(render_markdown(cell['source']))
            continue
        parts.append('<details><summary>Код Python</summary><pre>' + html.escape(cell['source']) + '</pre></details>')
        for out in cell.get('outputs', []):
            if out['output_type'] == 'stream':
                parts.append('<pre>'+html.escape(out['text'])+'</pre>')
                continue
            data = out.get('data', {})
            if 'text/html' in data:
                # Animation scripts and fixed element IDs are isolated from other cells.
                animation_number += 1
                match = re.search(r"show_animation\(.*?,\s*'([^']+\.html)'\)", cell['source'])
                animation_path = ROOT/'animations'/(match[1] if match else f'{path.stem}_{animation_number}.html')
                animation_path.parent.mkdir(exist_ok=True)
                animation_path.write_text('<!doctype html><meta charset="utf-8">'+data['text/html'], encoding='utf-8')
                parts.append('<iframe loading="lazy" src="animations/'+animation_path.name+'"></iframe>')
            elif 'image/png' in data:
                image_number += 1
                image_path = previews/f'{path.stem}_{image_number}.png'
                image_path.write_bytes(base64.b64decode(data['image/png']))
                parts.append('<div class="output"><img src="previews/'+image_path.name+'"></div>')
            elif 'text/markdown' in data:
                parts.append(render_markdown(data['text/markdown']))
            elif 'text/plain' in data:
                parts.append('<pre>'+html.escape(data['text/plain'])+'</pre>')
    parts.append('</html>')
    path.with_suffix('.html').write_text('\n'.join(parts), encoding='utf-8')


def execute(path):
    nb = json.loads(path.read_text(encoding='utf-8'))
    (ROOT/'.runtime'/'ipython').mkdir(parents=True, exist_ok=True)
    km = KernelManager(kernel_name='python3')
    km.start_kernel(cwd=str(ROOT), env={**os.environ, 'IPYTHONDIR': str(ROOT/'.runtime'/'ipython')},
                    extra_arguments=['--HistoryManager.hist_file=:memory:'])
    client = km.client()
    client.start_channels()
    client.wait_for_ready(timeout=60)
    start = time.perf_counter()
    count = 0
    try:
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] != 'code':
                continue
            count += 1
            cell['outputs'] = []
            msg_id = client.execute(cell['source'], stop_on_error=True)
            deadline = time.monotonic()+300
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError(f'{path.name} cell {i}')
                try:
                    msg = client.get_iopub_msg(timeout=1)
                except queue.Empty:
                    continue
                if msg['parent_header'].get('msg_id') != msg_id:
                    continue
                kind, content = msg['msg_type'], msg['content']
                if kind == 'execute_input':
                    cell['execution_count'] = content['execution_count']
                elif kind == 'stream':
                    cell['outputs'].append(dict(output_type='stream', name=content['name'], text=content['text']))
                elif kind in ('display_data', 'execute_result'):
                    out = dict(output_type=kind, data=content['data'], metadata=content['metadata'])
                    if kind == 'execute_result': out['execution_count'] = content['execution_count']
                    cell['outputs'].append(out)
                elif kind == 'error':
                    cell['outputs'].append(dict(output_type='error', ename=content['ename'],
                                                evalue=content['evalue'], traceback=content['traceback']))
                    raise RuntimeError('\n'.join(content['traceback']))
                elif kind == 'status' and content['execution_state'] == 'idle':
                    break
            print(f'{path.name}: code cell {count} OK ({time.perf_counter()-start:.1f}s)', flush=True)
    finally:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
        client.stop_channels()
        km.shutdown_kernel(now=True)
    export(path, nb)
    print(f'{path.name}: COMPLETE in {time.perf_counter()-start:.1f}s', flush=True)


if __name__ == '__main__':
    for path in ([Path(p).resolve() for p in sys.argv[1:]] if len(sys.argv)>1 else sorted(ROOT.glob('*.ipynb'))):
        execute(path)
