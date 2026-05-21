# Builds the three Task 1 report figures: pipeline diagram, preprocessing
# flowchart and BiLSTM architecture. Pure matplotlib so they render at A4
# without any external dependencies.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from src import paths


def add_box(ax, xy, w, h, label, fill='#E8F0FE', edge='#1A73E8', font=11, weight='bold'):
    box = mpatches.FancyBboxPatch(
        xy, w, h, boxstyle='round,pad=0.02',
        linewidth=1.4, edgecolor=edge, facecolor=fill,
    )
    ax.add_patch(box)
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    ax.text(cx, cy, label, ha='center', va='center', fontsize=font, weight=weight, wrap=True)


def add_arrow(ax, p0, p1, label=None, font=10):
    ax.annotate('', xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle='->', lw=1.6, color='#444'))
    if label:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx + 0.05, my + 0.05, label, fontsize=font, color='#444')


def make_pipeline_figure(out_path):
    fig, ax = plt.subplots(figsize=(12, 6.0))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.axis('off')

    add_box(ax, (0.2, 4.5), 1.9, 1.1, 'raw text\n(train / val / test)',
            fill='#FFF5E6', edge='#F29900')
    add_arrow(ax, (2.15, 5.05), (3.15, 5.05))

    add_box(ax, (3.2, 4.2), 2.6, 1.7,
            'Stage A — spam filter\nregex: line starts with\n"Subject:"  →  spam',
            fill='#E8F0FE', edge='#1A73E8')

    add_arrow(ax, (4.5, 4.15), (4.5, 2.15), label='spam')
    add_box(ax, (3.4, 1.05), 2.2, 1.0, 'dummy label  -1',
            fill='#FCE8E6', edge='#D93025')

    add_arrow(ax, (5.85, 5.05), (6.85, 5.05), label='review')

    stage_b_x = 6.9
    stage_b_y = 1.8
    stage_b_w = 5.7
    stage_b_h = 5.6
    add_box(ax, (stage_b_x, stage_b_y), stage_b_w, stage_b_h, '',
            fill='#F1F3F4', edge='#5F6368', font=12)
    ax.text(stage_b_x + stage_b_w / 2, stage_b_y + stage_b_h - 0.35,
            'Stage B — sentiment classifier',
            ha='center', va='center', fontsize=12, weight='bold')

    sub_y_top = stage_b_y + stage_b_h - 1.1
    sub_h = 0.7
    sub_pad = 0.22
    sub_x = stage_b_x + 0.25
    sub_w = stage_b_w - 0.5
    for i, (lbl, color) in enumerate([
        ('B1  TF-IDF (1-2 grams) + LogReg', '#E6F4EA'),
        ('B2  self-Word2Vec (200d) + LogReg', '#E6F4EA'),
        ('B3  pre-trained GloVe (100d) + LogReg', '#E6F4EA'),
        ('B4  BiLSTM (GloVe init, fine-tuned)  ★', '#FEF7E0'),
    ]):
        y = sub_y_top - i * (sub_h + sub_pad)
        add_box(ax, (sub_x, y - sub_h / 2), sub_w, sub_h, lbl,
                fill=color, edge='#34A853', font=10, weight='normal')

    output_y = stage_b_y + 0.45
    ax.text(stage_b_x + stage_b_w / 2, output_y, 'output: 0 / 1',
            ha='center', va='center',
            fontsize=11, weight='bold', color='#0F9D58',
            bbox=dict(facecolor='#E6F4EA', edgecolor='#0F9D58', boxstyle='round,pad=0.3'))

    add_arrow(ax, (5.6, 1.55), (stage_b_x + 0.5, output_y + 0.05))

    fig.suptitle('Figure 1: Two-stage pipeline (Task 1). Star marks the model used for the test submission.',
                 fontsize=11, y=0.04)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def make_bilstm_figure(out_path):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.axis('off')

    levels = [
        ('input tokens (max 400)', '#FFF5E6', '#F29900'),
        ('embedding layer  (vocab=8 606,  d=100)\nGloVe-initialised, fine-tuned', '#E8F0FE', '#1A73E8'),
        ('dropout  p=0.5', '#F1F3F4', '#5F6368'),
        ('BiLSTM  (1 layer, hidden=128 each direction)\npacked sequences, masked padding', '#FEF7E0', '#F9AB00'),
        ('concat( forward last hidden , backward last hidden )  →  256-d', '#E6F4EA', '#0F9D58'),
        ('dropout  p=0.5  →  Linear 256→64  →  ReLU', '#F1F3F4', '#5F6368'),
        ('dropout  p=0.5  →  Linear 64→1', '#F1F3F4', '#5F6368'),
        ('sigmoid  →  P(positive)', '#FCE8E6', '#D93025'),
    ]
    box_h = 0.85
    pad = 0.18
    y_start = 8.0
    box_w = 8.0
    box_x = 2.0

    coords = []
    for i, (label, fill, edge) in enumerate(levels):
        y = y_start - i * (box_h + pad)
        add_box(ax, (box_x, y - box_h / 2), box_w, box_h, label,
                fill=fill, edge=edge, font=11, weight='normal')
        coords.append((box_x + box_w / 2, y - box_h / 2, box_x + box_w / 2, y + box_h / 2))

    for i in range(len(coords) - 1):
        bottom_x, bottom_y, _, _ = coords[i]
        top_x, _, _, top_y = coords[i + 1]
        add_arrow(ax, (bottom_x, bottom_y), (top_x, top_y))

    side_text = (
        'Loss: BCEWithLogitsLoss\n'
        'Optimiser: Adam (lr 1e-3)\n'
        'Gradient clip: 5.0\n'
        'Batch size: 32\n'
        'Max epochs: 15\n'
        'Early stop: patience 2 on val loss\n'
        'Trainable params: ~1.11 M'
    )
    ax.text(11.0, 4.5, side_text, ha='left', va='center', fontsize=10,
            bbox=dict(facecolor='#FAFAFA', edgecolor='#5F6368', boxstyle='round,pad=0.4'))

    fig.suptitle('Figure 3: BiLSTM architecture (Task 1, B4).', fontsize=11, y=0.02)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def make_preprocess_figure(out_path):
    fig, ax = plt.subplots(figsize=(11, 4.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')

    steps = [
        'raw text',
        'lowercase',
        'word_tokenize\n(NLTK)',
        'drop punctuation\n(keep apostrophes)',
        'drop 1-char tokens\n(except "i", "a")',
        'drop English stopwords\nEXCEPT negations',
        'list of tokens',
    ]
    box_w = 1.55
    box_h = 1.4
    pad = 0.15
    total_w = len(steps) * box_w + (len(steps) - 1) * pad
    x_start = (12 - total_w) / 2

    centres = []
    for i, label in enumerate(steps):
        x = x_start + i * (box_w + pad)
        fill = '#FFF5E6' if i in (0, len(steps) - 1) else '#E8F0FE'
        edge = '#F29900' if i in (0, len(steps) - 1) else '#1A73E8'
        add_box(ax, (x, 2.0), box_w, box_h, label, fill=fill, edge=edge, font=9.5, weight='normal')
        centres.append((x + box_w / 2, 2.0 + box_h / 2))

    for i in range(len(centres) - 1):
        x0 = centres[i][0] + box_w / 2 - 0.05
        x1 = centres[i + 1][0] - box_w / 2 + 0.05
        y = 2.0 + box_h / 2
        add_arrow(ax, (x0, y), (x1, y))

    ax.text(6.0, 0.7,
            'Negation tokens preserved: not, no, never, n\'t, cannot, nor, none.\n'
            'Removing them inverts sentiment ("not good" → "good").',
            ha='center', va='center', fontsize=10, style='italic', color='#5F6368')

    fig.suptitle('Figure 2: Stage B pre-processing pipeline (Task 1).', fontsize=11, y=0.02)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    pipeline_path = paths.FIGURES / 'task1_pipeline.png'
    preprocess_path = paths.FIGURES / 'task1_preprocess.png'
    bilstm_path = paths.FIGURES / 'task1_bilstm_architecture.png'

    make_pipeline_figure(pipeline_path)
    make_preprocess_figure(preprocess_path)
    make_bilstm_figure(bilstm_path)

    print(f'wrote {pipeline_path}')
    print(f'wrote {preprocess_path}')
    print(f'wrote {bilstm_path}')


if __name__ == '__main__':
    main()
