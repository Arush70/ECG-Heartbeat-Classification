"""
Self-contained: 1D CNN architecture diagram for the ECG coursework.

Loads a real ECG beat directly from your training CSV (no helper files needed),
then draws a tensor-volume diagram showing the time axis contracting
(187 -> 93 -> 46 -> 23) while channel depth expands (1 -> 32 -> 64 -> 128),
with per-block shapes and parameter counts.

USAGE:
  1. Set DATA_PATH below to your mitbih_train.csv location.
  2. Run:  python make_architecture.py
  3. Outputs fig_architecture.png and fig_architecture.svg in the current folder.

Requires: numpy, matplotlib  (both already in your `ecg` environment)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, FancyBboxPatch

# ============================================================
#  EDIT THIS to point at your training CSV
# ============================================================
DATA_PATH = 'data/mitbih_train.csv'   # e.g. on Windows maybe 'data\\mitbih_train.csv'

# ---- Load one real Normal (class 0) ECG beat for the input visual ----
# We only read enough rows to find a class-0 beat (they are at the top of the file).
arr = np.loadtxt(DATA_PATH, delimiter=',', dtype=np.float32, max_rows=2000)
y = arr[:, -1].astype(int)
X = arr[:, :-1]
ecg = X[np.where(y == 0)[0][0]]   # first Normal beat

# ============================================================
#  Figure
# ============================================================
C_CONV = '#4A90D9'
C_CONV_TOP = '#6BA8E5'
C_CONV_SIDE = '#3576BE'
C_POOL = '#F4A259'
C_DENSE = '#8AB17D'
C_OUT = '#BC4749'
INK = '#1A1A2E'

fig = plt.figure(figsize=(16, 9))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 160)
ax.set_ylim(0, 90)
ax.axis('off')

def tensor_volume(x, y0, time_h, depth_n, color_face, color_top, color_side):
    """Pseudo-3D stack of slabs representing a (time, channels) tensor."""
    draw_n = min(depth_n, 14)
    dx, dy = 0.6, 0.6
    width = 7
    for i in range(draw_n - 1, -1, -1):
        ox, oy = i * dx, i * dy
        rect = FancyBboxPatch((x + ox, y0 + oy), width, time_h,
                              boxstyle="square,pad=0", linewidth=0.8,
                              edgecolor=INK, facecolor=color_face, alpha=0.92)
        ax.add_patch(rect)
    top = Polygon([(x, y0 + time_h), (x + width, y0 + time_h),
                   (x + width + (draw_n-1)*dx, y0 + time_h + (draw_n-1)*dy),
                   (x + (draw_n-1)*dx, y0 + time_h + (draw_n-1)*dy)],
                  closed=True, facecolor=color_top, edgecolor=INK, linewidth=0.8)
    ax.add_patch(top)
    side = Polygon([(x + width, y0),
                    (x + width + (draw_n-1)*dx, y0 + (draw_n-1)*dy),
                    (x + width + (draw_n-1)*dx, y0 + time_h + (draw_n-1)*dy),
                    (x + width, y0 + time_h)],
                   closed=True, facecolor=color_side, edgecolor=INK, linewidth=0.8)
    ax.add_patch(side)
    return x + width + (draw_n-1)*dx

def arrow(x0, y0, x1, y1, text=None, ty_off=3):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>',
                        mutation_scale=22, linewidth=2.2, color='#333')
    ax.add_patch(a)
    if text:
        ax.text((x0+x1)/2, max(y0, y1) + ty_off, text, ha='center', va='bottom',
                fontsize=9.5, color='#333', style='italic')

# Title
ax.text(80, 86, '1D CNN Architecture for ECG Heartbeat Classification',
        ha='center', fontsize=18, weight='bold', color=INK)
ax.text(80, 81.5, 'Time axis contracts (187 to 23) as channel depth expands (1 to 128); '
        'global pooling then a dense head map features to 5 class probabilities',
        ha='center', fontsize=10.5, style='italic', color='#555')

# Input ECG beat (inset axis)
ax_in = fig.add_axes([0.035, 0.42, 0.13, 0.20])
ax_in.plot(ecg, color=C_OUT, linewidth=1.6)
ax_in.set_title('Input ECG beat', fontsize=10, weight='bold')
ax_in.set_xlabel('time (187 samples)', fontsize=8)
ax_in.set_xticks([]); ax_in.set_yticks([])
for s in ax_in.spines.values():
    s.set_edgecolor(INK)

yc = 42
edge0 = 28

e1 = tensor_volume(30, yc, 26, 32, C_CONV, C_CONV_TOP, C_CONV_SIDE)
ax.text(30+3.5, yc-3.5, 'Block 1', ha='center', fontsize=10, weight='bold', color=INK)
ax.text(30+3.5, yc+30, '32 @ 93', ha='center', fontsize=9.5, color=INK, weight='bold')

e2 = tensor_volume(58, yc+3, 18, 64, C_CONV, C_CONV_TOP, C_CONV_SIDE)
ax.text(58+3.5, yc-0.5, 'Block 2', ha='center', fontsize=10, weight='bold', color=INK)
ax.text(58+3.5, yc+24, '64 @ 46', ha='center', fontsize=9.5, color=INK, weight='bold')

e3 = tensor_volume(86, yc+6, 11, 128, C_CONV, C_CONV_TOP, C_CONV_SIDE)
ax.text(86+3.5, yc+3, 'Block 3', ha='center', fontsize=10, weight='bold', color=INK)
ax.text(86+3.5, yc+20, '128 @ 23', ha='center', fontsize=9.5, color=INK, weight='bold')

arrow(edge0, yc+13, 30, yc+13, 'Conv k=7\n+BN+ReLU\n+Pool')
arrow(e1+2, yc+13, 58, yc+13, 'Conv k=7\n+Pool')
arrow(e2+2, yc+16, 86, yc+16, 'Conv k=7\n+Pool')

gx = 116
ax.add_patch(FancyBboxPatch((gx, yc+4), 3.5, 14, boxstyle="round,pad=0.1",
                            facecolor=C_POOL, edgecolor=INK, linewidth=1.2))
arrow(e3+2, yc+16, gx, yc+11, 'GAP')
ax.text(gx+1.7, yc+1, 'GAP\n128', ha='center', fontsize=9, weight='bold', color=INK)

dx0 = 126
ax.add_patch(FancyBboxPatch((dx0, yc+6), 3, 10, boxstyle="round,pad=0.1",
                            facecolor=C_DENSE, edgecolor=INK, linewidth=1.2))
arrow(gx+4, yc+11, dx0, yc+11, 'Dense\n+ReLU')
ax.text(dx0+1.5, yc+3, 'Dense\n64', ha='center', fontsize=9, weight='bold', color=INK)

ox = 138
for i, cl in enumerate(['N', 'S', 'V', 'F', 'Q']):
    ny = yc + 16 - i*2.6
    ax.add_patch(plt.Circle((ox, ny), 1.0, facecolor=C_OUT, edgecolor=INK, linewidth=1.2))
    ax.text(ox, ny, cl, ha='center', va='center', fontsize=8, color='white', weight='bold')
arrow(dx0+3.5, yc+11, ox-1.5, yc+11, 'Dense\nSoftmax')
ax.text(ox+4.5, yc+11, '5 class\nprobabilities', ha='left', va='center',
        fontsize=9.5, weight='bold', color=INK)

# Detail strip
detail_y = 14
details = [
    ('INPUT', '(187, 1)', '0 params', 'Single-channel\nECG waveform', C_OUT),
    ('CONV BLOCK 1', '(93, 32)', '256 params', 'Conv1D(32,k=7)\nBN+ReLU+Pool+Drop', C_CONV),
    ('CONV BLOCK 2', '(46, 64)', '14,400 params', 'Conv1D(64,k=7)\nBN+ReLU+Pool+Drop', C_CONV),
    ('CONV BLOCK 3', '(23, 128)', '57,472 params', 'Conv1D(128,k=7)\nBN+ReLU+Pool+Drop', C_CONV),
    ('GLOBAL AVG POOL', '(128)', '0 params', 'Collapse time axis\nby averaging', C_POOL),
    ('DENSE', '(64)', '8,256 params', 'Fully connected\n+ReLU +Dropout', C_DENSE),
    ('OUTPUT', '(5)', '325 params', 'Softmax over\n5 beat classes', C_OUT),
]
box_w = 20.5
for i, (name, shape, params, desc, col) in enumerate(details):
    bx = 4 + i*22
    ax.add_patch(FancyBboxPatch((bx, detail_y), box_w, 11,
                 boxstyle="round,pad=0.15,rounding_size=0.4",
                 facecolor='white', edgecolor=col, linewidth=2.2))
    ax.add_patch(FancyBboxPatch((bx, detail_y+8.2), box_w, 2.8,
                 boxstyle="round,pad=0.15,rounding_size=0.4",
                 facecolor=col, edgecolor=col, linewidth=1))
    ax.text(bx+box_w/2, detail_y+9.6, name, ha='center', va='center',
            fontsize=8, weight='bold', color='white')
    ax.text(bx+box_w/2, detail_y+6.3, shape, ha='center', va='center',
            fontsize=10, weight='bold', color=INK, family='monospace')
    ax.text(bx+box_w/2, detail_y+4.2, params, ha='center', va='center',
            fontsize=7.5, color='#666', style='italic')
    ax.text(bx+box_w/2, detail_y+1.8, desc, ha='center', va='center',
            fontsize=7, color='#333', linespacing=1.3)
    if i < len(details)-1:
        ax.annotate('', xy=(bx+box_w+1.5, detail_y+5.5), xytext=(bx+box_w, detail_y+5.5),
                    arrowprops=dict(arrowstyle='-|>', color='#999', lw=1.5))

ax.text(80, 8.5, 'Total trainable parameters: 81,605   |   Optimiser: Adam   |   '
        'Loss: class-weighted categorical cross-entropy',
        ha='center', fontsize=10, weight='bold', color=INK)

plt.savefig('fig_architecture.png', bbox_inches='tight', dpi=200, facecolor='white')
plt.savefig('fig_architecture.svg', bbox_inches='tight', facecolor='white')
print('Saved fig_architecture.png and fig_architecture.svg')
