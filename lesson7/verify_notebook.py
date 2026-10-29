"""Выполнить ноутбук с чистого ядра, сохранить выводы и PNG для визуальной проверки."""
from pathlib import Path
import base64
import os
import sys
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parent
path = root / (sys.argv[1] if len(sys.argv)>1 else 'lesson_07_heat.ipynb')
qa = root / '_qa' / path.stem
qa.mkdir(exist_ok=True, parents=True)
os.environ['MPLCONFIGDIR'] = str(qa / 'matplotlib')
os.environ['IPYTHONDIR'] = str(qa / 'ipython')
os.environ['JUPYTER_RUNTIME_DIR'] = str(qa / 'runtime')
nb = nbformat.read(path, as_version=4)
client = NotebookClient(nb, timeout=180, kernel_name='python3',
                        resources={'metadata': {'path': str(root)}})
client.execute()
nbformat.validate(nb)
nbformat.write(nb, path)
images = 0
for i, cell in enumerate(nb.cells):
    for output in cell.get('outputs', []):
        if output.output_type == 'error':
            raise RuntimeError(output)
        if output.output_type == 'stream':
            print(output.text.strip())
        png = output.get('data', {}).get('image/png')
        if png:
            images += 1
            (qa / f'figure_{images:02d}.png').write_bytes(base64.b64decode(png))
print(f'Выполнено {sum(c.cell_type == "code" for c in nb.cells)} кодовых ячеек; PNG: {images}')
