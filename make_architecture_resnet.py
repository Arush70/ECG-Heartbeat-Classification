"""
Self-contained: Residual 1D CNN architecture diagram for the ECG coursework.

Shows the residual (skip) connections explicitly, plus the tensor-volume flow
(time 187 -> 93 -> 46 -> 23, channels 1 -> 32 -> 64 -> 128) and per-block details.

USAGE: set DATA_PATH, then `python make_architecture_resnet.py`
Requires: numpy, matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, FancyBboxPatch

DATA_PATH = 'data/mitbih_train.csv'   # EDIT to your path

# Real Normal beat for the input visual
arr = np.loadtxt(DATA_PATH, delimiter=',', dtype=np.float32, max_rows=2000)
y = arr[:, -1].astype(int); X = arr[:, :-1]
ecg = X[np.where(y == 0)[0][0]]

C_CONV = '#4A90D9'; C_CONV_TOP = '#6BA8E5'; C_CONV_SIDE = '#3576BE'
C_POOL = '#F4A259'; C_DENSE = '#8AB17D'; C_OUT = '#BC4749'
C_SKIP = '#9B5DE5'; INK = '#1A1A2E'

fig = plt.figure(figsize=(16, 9.5))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 160); ax.set_ylim(0, 95); ax.axis('off')

def tensor_volume(x, y0, time_h, depth_n, cf, ct, cs):
    draw_n = min(depth_n, 14); dx, dy = 0.6, 0.6; width = 7
    for i in range(draw_n - 1, -1, -1):
        ox, oy = i*dx, i*dy
        ax.add_patch(FancyBboxPatch((x+ox, y0+oy), width, time_h,
                     boxstyle="square,pad=0", linewidth=0.8,
                     edgecolor=INK, facecolor=cf, alpha=0.92))
    ax.add_patch(Polygon([(x, y0+time_h), (x+width, y0+time_h),
                   (x+width+(draw_n-1)*dx, y0+time_h+(draw_n-1)*dy),
                   (x+(draw_n-1)*dx, y0+time_h+(draw_n-1)*dy)],
                  closed=True, facecolor=ct, edgecolor=INK, linewidth=0.8))
    ax.add_patch(Polygon([(x+width, y0),
                    (x+width+(draw_n-1)*dx, y0+(draw_n-1)*dy),
                    (x+width+(draw_n-1)*dx, y0+time_h+(draw_n-1)*dy),
                    (x+width, y0+time_h)],
                   closed=True, facecolor=cs, edgecolor=INK, linewidth=0.8))
    return x+width+(draw_n-1)*dx

def arrow(x0, y0, x1, y1, text=None, ty=3, color='#333'):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>',
                 mutation_scale=22, linewidth=2.2, color=color))
    if text:
        ax.text((x0+x1)/2, max(y0, y1)+ty, text, ha='center', va='bottom',
                fontsize=9, color=color, style='italic')

def skip_arc(x0, x1, ybase):
    """Draw a curved skip connection above the blocks."""
    mid = (x0+x1)/2
    ax.add_patch(FancyArrowPatch((x0, ybase), (x1, ybase),
                 connectionstyle="arc3,rad=-0.45", arrowstyle='-|>',
                 mutation_scale=16, linewidth=2.0, color=C_SKIP, linestyle='--'))
    ax.text(mid, ybase+6.0, 'skip (+)', ha='center', fontsize=8.5,
            color=C_SKIP, weight='bold', style='italic')

# Title
ax.text(80, 91, 'Residual 1D CNN Architecture for ECG Heartbeat Classification',
        ha='center', fontsize=17, weight='bold', color=INK)
ax.text(80, 86.8, 'Three residual blocks with skip connections; time axis contracts '
        '(187 to 23) as channel depth expands (1 to 128)',
        ha='center', fontsize=10.5, style='italic', color='#555')

# Input ECG
ax_in = fig.add_axes([0.035, 0.43, 0.12, 0.18])
ax_in.plot(ecg, color=C_OUT, linewidth=1.6)
ax_in.set_title('Input ECG beat', fontsize=10, weight='bold')
ax_in.set_xlabel('time (187 samples)', fontsize=8)
ax_in.set_xticks([]); ax_in.set_yticks([])
for s in ax_in.spines.values(): s.set_edgecolor(INK)

yc = 40; edge0 = 27

e1 = tensor_volume(29, yc, 26, 32, C_CONV, C_CONV_TOP, C_CONV_SIDE)
ax.text(29+3.5, yc-3.5, 'Res Block 1', ha='center', fontsize=9.5, weight='bold', color=INK)
ax.text(29+3.5, yc+30, '32 @ 93', ha='center', fontsize=9.5, color=INK, weight='bold')

e2 = tensor_volume(57, yc+3, 18, 64, C_CONV, C_CONV_TOP, C_CONV_SIDE)
ax.text(57+3.5, yc-0.5, 'Res Block 2', ha='center', fontsize=9.5, weight='bold', color=INK)
ax.text(57+3.5, yc+24, '64 @ 46', ha='center', fontsize=9.5, color=INK, weight='bold')

e3 = tensor_volume(85, yc+6, 11, 128, C_CONV, C_CONV_TOP, C_CONV_SIDE)
ax.text(85+3.5, yc+3, 'Res Block 3', ha='center', fontsize=9.5, weight='bold', color=INK)
ax.text(85+3.5, yc+20, '128 @ 23', ha='center', fontsize=9.5, color=INK, weight='bold')

arrow(edge0, yc+13, 29, yc+13)
arrow(e1+2, yc+13, 57, yc+13)
arrow(e2+2, yc+16, 85, yc+16)

# Skip-connection arcs over each block
skip_arc(30, 37, yc+34)
skip_arc(58, 65, yc+35)
skip_arc(86, 93, yc+36)

gx = 115
ax.add_patch(FancyBboxPatch((gx, yc+4), 3.5, 14, boxstyle="round,pad=0.1",
             facecolor=C_POOL, edgecolor=INK, linewidth=1.2))
arrow(e3+2, yc+16, gx, yc+11)
ax.text(gx+1.7, yc+1, 'GAP\n128', ha='center', fontsize=9, weight='bold', color=INK)

dx0 = 125
ax.add_patch(FancyBboxPatch((dx0, yc+6), 3, 10, boxstyle="round,pad=0.1",
             facecolor=C_DENSE, edgecolor=INK, linewidth=1.2))
arrow(gx+4, yc+11, dx0, yc+11)
ax.text(dx0+1.5, yc+3, 'Dense\n64', ha='center', fontsize=9, weight='bold', color=INK)

ox = 137
for i, cl in enumerate(['1', '2', '3', '4', '5']):
    ny = yc + 16 - i*2.6
    ax.add_patch(plt.Circle((ox, ny), 1.0, facecolor=C_OUT, edgecolor=INK, linewidth=1.2))
    ax.text(ox, ny, cl, ha='center', va='center', fontsize=8, color='white', weight='bold')
arrow(dx0+3.5, yc+11, ox-1.5, yc+11)
ax.text(ox+4.5, yc+11, '5 class\nprobabilities', ha='left', va='center',
        fontsize=9.5, weight='bold', color=INK)

# Inset: what one residual block contains
ix, iy, iw, ih = 112, 60, 44, 22
ax.add_patch(FancyBboxPatch((ix, iy), iw, ih, boxstyle="round,pad=0.3,rounding_size=0.8",
             facecolor='#F7F7FB', edgecolor=C_SKIP, linewidth=2))
ax.text(ix+iw/2, iy+ih-2.5, 'Inside one residual block', ha='center',
        fontsize=9.5, weight='bold', color=C_SKIP)
steps = ['Conv1D(k=7) - BN - ReLU', 'Conv1D(k=7) - BN', '(+) add input (skip)',
         'ReLU - MaxPool(2) - Dropout']
for i, s in enumerate(steps):
    ax.text(ix+iw/2, iy+ih-6-i*4, s, ha='center', fontsize=8.2, color=INK)

# Detail strip
detail_y = 13
details = [
    ('INPUT', '(187, 1)', '0 params', 'Single-channel\nECG waveform', C_OUT),
    ('RES BLOCK 1', '(93, 32)', '~8.7k params', '2x Conv1D(32)\n+skip +Pool', C_CONV),
    ('RES BLOCK 2', '(46, 64)', '~44k params', '2x Conv1D(64)\n+skip +Pool', C_CONV),
    ('RES BLOCK 3', '(23, 128)', '~173k params', '2x Conv1D(128)\n+skip +Pool', C_CONV),
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
            fontsize=7.6, weight='bold', color='white')
    ax.text(bx+box_w/2, detail_y+6.3, shape, ha='center', va='center',
            fontsize=10, weight='bold', color=INK, family='monospace')
    ax.text(bx+box_w/2, detail_y+4.2, params, ha='center', va='center',
            fontsize=7.3, color='#666', style='italic')
    ax.text(bx+box_w/2, detail_y+1.8, desc, ha='center', va='center',
            fontsize=6.8, color='#333', linespacing=1.3)
    if i < len(details)-1:
        ax.annotate('', xy=(bx+box_w+1.5, detail_y+5.5), xytext=(bx+box_w, detail_y+5.5),
                    arrowprops=dict(arrowstyle='-|>', color='#999', lw=1.5))

ax.text(80, 7.5, 'Total trainable parameters: 244,645   |   Optimiser: Adam   |   '
        'Loss: categorical cross-entropy (no class weighting)',
        ha='center', fontsize=10, weight='bold', color=INK)

plt.savefig('fig_architecture.png', bbox_inches='tight', dpi=200, facecolor='white')
plt.savefig('fig_architecture.svg', bbox_inches='tight', facecolor='white')
print('Saved fig_architecture.png and fig_architecture.svg')
