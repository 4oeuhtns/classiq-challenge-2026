import sys
import pathlib

ROOT = next(p for p in [pathlib.Path(__file__).resolve(), *pathlib.Path(__file__).resolve().parents]
            if (p / "oracle").is_dir())
sys.path.insert(0, str(ROOT))

LOGO = ROOT / "cpp" / "logo.pla"

from oracle.mask import logo_pixel

lines = ['.i 12', '.o 1']
for y in range(64):
    for x in range(64):
        if logo_pixel(x, y):
            bits = ''.join(str((x >> b) & 1)
for b in range(6)) + \
                    ''.join(str((y >> b) & 1)
for b in range(6))
            lines.append(bits + ' 1')
lines.append('.e')

with open(LOGO, 'w') as f:
    f.write('\n'.join(lines) + '\n')
print("wrote", len(lines) - 3, "rows")