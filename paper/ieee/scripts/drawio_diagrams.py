"""Editable draw.io sources and deterministic vector exports for the paper.

Sources use native mxGraph shapes and orthogonal edges, not embedded bitmaps.
Run this script to export existing sources; delete a source only to reset its design.
All flowchart boxes use one light brown-cream fill with a brown outline (FILL, STROKE).
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'diagrams'
SOURCE.mkdir(exist_ok=True)

FILL = '#F3E7D3'      # light brown-cream, the single box colour of every flowchart
STROKE = '#8A6A45'    # brown outline
INK = '#3B2A17'       # dark brown text and arrows


class Diagram:
    def __init__(self, name, width, height):
        self.name, self.width, self.height = name, width, height
        self.doc = ET.Element('mxfile', host='app.diagrams.net')
        page = ET.SubElement(self.doc, 'diagram', name=name, id=name)
        graph = ET.SubElement(page, 'mxGraphModel', page='1', pageWidth=str(width), pageHeight=str(height),
                              grid='1', gridSize='10')
        self.root = ET.SubElement(graph, 'root')
        ET.SubElement(self.root, 'mxCell', id='0')
        ET.SubElement(self.root, 'mxCell', id='1', parent='0')

    def box(self, key, x, y, w, h, label, dashed=False, bold=False, text=False):
        style = (f'rounded=1;arcSize=12;whiteSpace=wrap;html=0;fillColor={FILL};strokeColor={STROKE};'
                 f'fontColor={INK};fontFamily=Times New Roman;fontSize=12.5;strokeWidth=1;'
                 f'dashed={int(dashed)};fontStyle={int(bold)};')
        if text:
            style += 'shape=text;fillColor=none;strokeColor=none;align=left;'
        cell = ET.SubElement(self.root, 'mxCell', id=key, value=label, style=style, vertex='1', parent='1')
        ET.SubElement(cell, 'mxGeometry', x=str(x), y=str(y), width=str(w), height=str(h), attrib={'as': 'geometry'})

    def edge(self, key, points, dashed=False):
        cell = ET.SubElement(self.root, 'mxCell', id=key,
                             style=f'edgeStyle=orthogonalEdgeStyle;rounded=0;html=0;endArrow=block;endFill=1;'
                                   f'endSize=10;strokeColor={INK};strokeWidth=1.6;dashed={int(dashed)};',
                             edge='1', parent='1')
        geo = ET.SubElement(cell, 'mxGeometry', relative='1', attrib={'as': 'geometry'})
        for tag, p in [('sourcePoint', points[0]), ('targetPoint', points[-1])]:
            ET.SubElement(geo, 'mxPoint', x=str(p[0]), y=str(p[1]), attrib={'as': tag})
        if len(points) > 2:
            array = ET.SubElement(geo, 'Array', attrib={'as': 'points'})
            for p in points[1:-1]:
                ET.SubElement(array, 'mxPoint', x=str(p[0]), y=str(p[1]))

    def write(self):
        path = SOURCE / (self.name + '.drawio')
        if not path.exists():
            ET.ElementTree(self.doc).write(path, encoding='utf-8', xml_declaration=True)


def defaults():
    # Model: graph encoder, read-out, non-SEIR or SEIR heads, persistence anchor, SEIR inputs.
    d = Diagram('architecture', 716, 235)
    d.box('input', 8, 16, 122, 66, 'Past case counts\nlog(1 + c), scaled with\ntraining statistics')
    d.box('encoder', 153, 16, 126, 66, 'Graph encoder\nstatic or adaptive\nadjacency')
    d.box('readout', 302, 16, 98, 66, 'Read-out\nper district\nand horizon')
    d.box('nonseir', 423, 12, 140, 66, 'Non-SEIR heads\ndirect / residual / gated')
    d.box('seirhead', 423, 106, 140, 66, 'SEIR heads, bare or gated\nanchored force of infection\nS → E → I → R, daily')
    d.box('output', 586, 32, 122, 76, 'Weekly forecasts\nh = 1, 2, 3\n(cases)')
    for k, a, b in [('e1', 130, 153), ('e2', 279, 302), ('e3', 400, 423)]:
        d.edge(k, [(a, 45), (b, 45)])
    d.edge('branch', [(400, 45), (410, 45), (410, 122), (423, 122)])
    d.edge('nonseir_out', [(563, 45), (574, 45), (574, 58), (586, 58)])
    d.edge('seir_out', [(563, 139), (574, 139), (574, 92), (586, 92)])
    d.box('anchor', 8, 131, 185, 60, 'Last observed week\npersistence anchor for\nresidual and gated heads')
    d.box('initial', 229, 131, 150, 60, 'District population +\ninitial S, E, I, R\nfrom recent counts')
    d.edge('anchor_nonseir', [(193, 160), (211, 160), (211, 93), (463, 93), (463, 78)])
    d.edge('anchor_seir', [(193, 160), (211, 160), (211, 203), (493, 203), (493, 172)])
    d.edge('initial_seir', [(379, 160), (423, 160)])
    d.box('loss', 586, 144, 122, 47, 'Training: squared\nerror on scaled targets', dashed=True)
    d.edge('loss_edge', [(647, 108), (647, 144)], True)
    d.box('legend', 8, 213, 700, 19,
          'Solid arrows: forecasting path     Dashed: training only     SEIR path: daily compartment flows', text=True)
    d.write()

    # Evaluation: rolling origins, splits, seeds and paired tests.
    d = Diagram('evaluation_protocol', 350, 192)
    d.box('series', 5, 8, 150, 40, 'Rebuilt series\n559 weeks, 25 districts')
    d.box('origins', 195, 8, 150, 40, 'Nine rolling origins\n0.50, 0.55, ..., 0.90')
    d.edge('o1', [(155, 28), (195, 28)])
    d.box('train', 5, 68, 100, 50, 'Training\nall earlier windows')
    d.box('val', 125, 68, 100, 50, 'Validation\n30 windows,\nearly stopping')
    d.box('test', 245, 68, 100, 50, 'Test\nnext 5%\nof windows')
    d.edge('o2', [(270, 48), (270, 57), (55, 57), (55, 68)])
    d.edge('t1', [(105, 93), (125, 93)])
    d.edge('t2', [(225, 93), (245, 93)])
    d.box('seeds', 5, 146, 150, 40, 'Seeds 0, 1, 2\naveraged per origin')
    d.box('stats', 195, 146, 150, 40, 'Paired sign-flip test\nagainst persistence')
    d.edge('o3', [(295, 118), (295, 132), (80, 132), (80, 146)])
    d.edge('s1', [(155, 166), (195, 166)])
    d.write()


def export(path, output):
    tree = ET.parse(path)
    graph = tree.find('.//mxGraphModel')
    width, height = float(graph.get('pageWidth')), float(graph.get('pageHeight'))
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0, width), ylim=(height, 0))
    ax.axis('off')
    for cell in tree.findall('.//mxCell[@vertex="1"]'):
        st = dict(pair.split('=', 1) for pair in cell.get('style', '').split(';') if '=' in pair)
        g = cell.find('mxGeometry')
        x, y, w, h = [float(g.get(k)) for k in ('x', 'y', 'width', 'height')]
        text = st.get('shape') == 'text'
        if not text:
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=5',
                                        facecolor=st['fillColor'], edgecolor=st['strokeColor'], lw=.8,
                                        linestyle='--' if st.get('dashed') == '1' else '-', zorder=2))
        ax.text(x if text else x + w / 2, y + h / 2, cell.get('value', ''), ha='left' if text else 'center',
                va='center', fontsize=float(st.get('fontSize', 12.5)) * .72,
                fontfamily=st.get('fontFamily', 'Times New Roman'),
                fontweight='bold' if st.get('fontStyle') == '1' else 'normal',
                color=st.get('fontColor', INK), zorder=3)
    for cell in tree.findall('.//mxCell[@edge="1"]'):
        st = dict(pair.split('=', 1) for pair in cell.get('style', '').split(';') if '=' in pair)
        colour = st.get('strokeColor', INK)
        g = cell.find('mxGeometry')
        points = [(float(p.get('x')), float(p.get('y'))) for p in
                  [g.find("mxPoint[@as='sourcePoint']"), *g.findall('Array/mxPoint'),
                   g.find("mxPoint[@as='targetPoint']")]]
        ls = '--' if st.get('dashed') == '1' else '-'
        for a, b in zip(points[:-2], points[1:-1]):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=colour, lw=1.15, ls=ls, zorder=4)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>', mutation_scale=11, lw=1.15,
                                     linestyle=ls, color=colour, shrinkA=0, shrinkB=0, zorder=4))
    plt.rcParams['pdf.fonttype'] = 42
    fig.savefig(ROOT / 'figures' / output, metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)


def architecture():
    defaults()
    export(SOURCE / 'architecture.drawio', 'fig_arch.pdf')


def protocol():
    defaults()
    export(SOURCE / 'evaluation_protocol.drawio', 'evaluation_protocol.pdf')


if __name__ == '__main__':
    architecture()
    protocol()
