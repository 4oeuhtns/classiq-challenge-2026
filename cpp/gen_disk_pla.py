"""Reduce a disk to its LOCAL bounding-box truth table (plan.md Strategy D)
and write it as a .pla mockturtle can read. Local coordinates are just the
low bits of x,y within the box -- no subtraction needed, per plan.md's
'bounding box, then low bits' trick (II.5).
"""
import sys

DISKS = {
    "disk1": (lambda x, y: (x - 40) ** 2 + (y - 19) ** 2 <= 72, 32, 11, 5),
    "disk2": (lambda x, y: (x - 55) ** 2 + (y - 41) ** 2 <= 42, 49, 35, 4),
}

name = sys.argv[1] if len(sys.argv) > 1 else "disk1"
pixel_fn, box_x0, box_y0, bits = DISKS[name]
size = 1 << bits

lines = [f".i {bits * 2}", ".o 1"]
n_px = 0
for yl in range(size):
    for xl in range(size):
        x, y = box_x0 + xl, box_y0 + yl
        if 0 <= x < 64 and 0 <= y < 64 and pixel_fn(x, y):
            row = ''.join(str((xl >> b) & 1) for b in range(bits)) + \
                  ''.join(str((yl >> b) & 1) for b in range(bits))
            lines.append(row + ' 1')
            n_px += 1
lines.append(".e")

out_path = f"{name}.pla"
with open(out_path, "w") as f:
    f.write('\n'.join(lines) + '\n')
print(f"wrote {out_path}: {bits*2} vars, {n_px} true rows out of {size*size}")
