"""由已保存数值轨迹生成XML安全的独立SVG，不改变权重、训练代码或原轨迹。"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', NS)


def render(curve, best_epoch, title, path):
    def element(parent, kind, text=None, **attrs):
        node = ET.SubElement(parent, '{' + NS + '}' + kind, {k.replace('_', '-'): str(v) for k, v in attrs.items()})
        if text is not None:
            node.text = text
        return node
    root = ET.Element('{' + NS + '}svg', {'width': '900', 'height': '610', 'viewBox': '0 0 900 610'})
    element(root, 'rect', width='100%', height='100%', fill='white')
    element(root, 'style', text='text { font-family: sans-serif; font-size: 12px; }')
    element(root, 'text', text=title, x=60, y=25, font_size=16)
    last = curve[-1]['epoch']
    for top, height, series, caption in [
        (70, 215, [('train_bce', '#1768ac'), ('val_bce', '#d1495b')], 'Train / validation BCE'),
        (365, 195, [('val_A_recall', '#16845b'), ('val_A_fpr', '#a65c00')], 'Validation working point: A FPR <= 5%')]:
        left, width = 65, 790
        maximum = max(.1, 1.05 * max(r[key] for r in curve for key, _ in series))
        element(root, 'text', text=caption, x=left, y=top - 25)
        for step in range(5):
            y = top + height * step / 4
            element(root, 'line', x1=left, y1=y, x2=left + width, y2=y, stroke='#ddd')
            element(root, 'text', text=f'{maximum * (1 - step / 4):.3f}', x=5, y=y + 4)
        for epoch in range(0, last + 1, max(1, last // 5)):
            x = left + width * epoch / last
            element(root, 'text', text=str(epoch), x=x, y=top + height + 20)
        xb = left + width * best_epoch / last
        element(root, 'line', x1=xb, y1=top, x2=xb, y2=top + height, stroke='#777', stroke_dasharray='4 4')
        for i, (key, color) in enumerate(series):
            points = ' '.join(f'{left + width * r["epoch"] / last:.2f},{top + height * (1-r[key]/maximum):.2f}' for r in curve)
            element(root, 'polyline', points=points, fill='none', stroke=color, stroke_width=1.8)
            element(root, 'text', text=key, x=left + 470 + 150 * i, y=top - 7, fill=color)
    element(root, 'text', text=f'Dashed line: selected epoch {best_epoch}; test is not used for checkpoint selection.', x=65, y=600)
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)
    ET.parse(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir')
    args = parser.parse_args()
    run = Path(args.run_dir)
    generated = []
    for arm in ['S1', 'S2']:
        for name in ['G', 'E']:
            for seed in [42, 43, 44]:
                folder = run / arm / f'{name}-seed-{seed}'
                path = folder / 'trajectory.checked.svg'
                if path.exists():
                    raise ValueError('不覆盖已生成的checked曲线')
                curve = [json.loads(line) for line in (folder / 'trajectory.jsonl').open()]
                summary = json.loads((folder / 'summary.json').read_text())
                render(curve, summary['best_epoch'], f'{arm} / {name} / seed {seed}', path)
                generated.append(str(path.relative_to(run)))
    (run / 'visualization_repair.json').write_text(json.dumps({
        'reason': '旧SVG标题中未转义的<导致XML无效；仅重绘展示层',
        'original_numeric_trajectories_unchanged': True,
        'new_files': generated, 'preferred_plot': 'trajectory.checked.svg'}, ensure_ascii=False, indent=2))
    print(json.dumps({'validated_svg_files': len(generated)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
