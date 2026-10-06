"""Editable draw.io sources and deterministic vector exports for the paper.

Sources use native mxGraph shapes and orthogonal edges, not embedded bitmaps.
Run this script to export existing sources; delete a source only to reset its design.
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

class Diagram:
    def __init__(self, name, width, height):
        self.name, self.width, self.height = name, width, height
        self.doc=ET.Element('mxfile',host='app.diagrams.net')
        page=ET.SubElement(self.doc,'diagram',name=name,id=name)
        graph=ET.SubElement(page,'mxGraphModel',page='1',pageWidth=str(width),pageHeight=str(height),grid='1',gridSize='10')
        self.root=ET.SubElement(graph,'root')
        ET.SubElement(self.root,'mxCell',id='0')
        ET.SubElement(self.root,'mxCell',id='1',parent='0')
    def box(self, key, x,y,w,h,label,fill='#EEF4FA',stroke='#35546F',dashed=False,bold=False,text=False):
        style=f'rounded=1;arcSize=12;whiteSpace=wrap;html=0;fillColor={fill};strokeColor={stroke};fontColor=#172B3A;fontFamily=Times New Roman;fontSize=12.5;strokeWidth=1;dashed={int(dashed)};fontStyle={int(bold)};'
        if text: style+='shape=text;fillColor=none;strokeColor=none;align=left;'
        cell=ET.SubElement(self.root,'mxCell',id=key,value=label,style=style,vertex='1',parent='1')
        ET.SubElement(cell,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'})
    def edge(self,key,points,dashed=False):
        cell=ET.SubElement(self.root,'mxCell',id=key,style=f'edgeStyle=orthogonalEdgeStyle;rounded=0;html=0;endArrow=block;endFill=1;endSize=10;strokeColor=#202020;strokeWidth=1.6;dashed={int(dashed)};',edge='1',parent='1')
        geo=ET.SubElement(cell,'mxGeometry',relative='1',attrib={'as':'geometry'})
        for tag,p in [('sourcePoint',points[0]),('targetPoint',points[-1])]: ET.SubElement(geo,'mxPoint',x=str(p[0]),y=str(p[1]),attrib={'as':tag})
        if len(points)>2:
            array=ET.SubElement(geo,'Array',attrib={'as':'points'})
            for p in points[1:-1]:ET.SubElement(array,'mxPoint',x=str(p[0]),y=str(p[1]))
    def write(self):
        path=SOURCE/(self.name+'.drawio')
        if not path.exists():ET.ElementTree(self.doc).write(path,encoding='utf-8',xml_declaration=True)

def defaults():
    d=Diagram('architecture',716,255)
    d.box('title',8,5,600,22,'Historical forecasting paths',text=True,bold=True)
    d.box('input',8,36,122,66,'Past case counts\nLog transform +\ntraining-scale statistics')
    d.box('encoder',153,36,126,66,'Graph encoder\nStatic / adaptive\nadjacency')
    d.box('readout',302,36,98,66,'Read-out\nPer horizon')
    d.box('nonseir',423,32,140,66,'Non-SEIR heads\nDirect / residual / gated')
    d.box('seirhead',423,126,140,66,'SEIR / gated SEIR\nForce of infection\nS \u2192 E \u2192 I \u2192 R',fill='#F4F1E7')
    d.box('output',586,52,122,76,'Forecast output\nLog scale / counts',fill='#E9F3EE')
    for k,a,b in [('e1',130,153),('e2',279,302),('e3',400,423)]:d.edge(k,[(a,65),(b,65)])
    d.edge('branch',[(400,65),(410,65),(410,142),(423,142)])
    d.edge('nonseir_out',[(563,65),(574,65),(574,78),(586,78)])
    d.edge('seir_out',[(563,159),(574,159),(574,112),(586,112)])
    d.box('anchor',8,151,185,60,'Last observed cases\nPersistence anchor\nResidual / gated heads',fill='#F4F1E7')
    d.box('initial',229,151,150,60,'Population +\ninitial compartments',fill='#F4F1E7')
    d.edge('anchor_nonseir',[(193,180),(211,180),(211,113),(463,113),(463,98)])
    d.edge('anchor_seir',[(193,180),(211,180),(211,223),(493,223),(493,192)])
    d.edge('initial_seir',[(379,180),(423,180)])
    d.box('loss',586,164,122,47,'Training only\nTargets + squared loss',fill='#F4F4F4',stroke='#666666',dashed=True)
    d.edge('loss_edge',[(647,128),(647,164)],True)
    d.box('legend',8,232,700,19,'Solid arrows: inference     Dashed arrow: training only     SEIR path: daily compartment flows',text=True)
    d.write()
    d=Diagram('evaluation_protocol',350,200)
    d.box('title',5,2,340,21,'Observed: historical rolling origins',text=True,bold=True)
    for key,x,w,label,fill in [('train',5,90,'Training','#E8F0F7'),('val',121,92,'Validation','#F4F1E7'),('dev',237,108,'Development\nevaluation','#E9F3EE')]:d.box(key,x,32,w,42,label,fill)
    d.edge('v1',[(95,53),(121,53)]);d.edge('v2',[(213,53),(237,53)])
    d.box('overlap',5,79,340,32,'Forecast starts are chronological.\nTwo target weeks overlap at each boundary.',text=True)
    d.box('recommend',5,114,340,17,'Recommended after correcting the splits',text=True,bold=True)
    d.box('freeze',5,138,124,38,'Select on validation\nFreeze all settings',fill='#F4F1E7')
    d.box('confirm',158,138,187,38,'Untouched confirmation\nNo further tuning',fill='#F4F4F4',stroke='#666666',dashed=True)
    d.edge('future',[(129,157),(158,157)],True)
    d.box('footer',5,179,340,19,'Recommendation only; no confirmation result reported.',text=True)
    d.write()

def export(path,output):
    tree=ET.parse(path);graph=tree.find('.//mxGraphModel')
    width,height=float(graph.get('pageWidth')),float(graph.get('pageHeight'))
    fig,ax=plt.subplots(figsize=(width/100,height/100))
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.set(xlim=(0,width),ylim=(height,0));ax.axis('off')
    for cell in tree.findall('.//mxCell[@vertex="1"]'):
        st=dict(pair.split('=',1) for pair in cell.get('style','').split(';') if '=' in pair)
        g=cell.find('mxGeometry');x,y,w,h=[float(g.get(k)) for k in ('x','y','width','height')]
        text=st.get('shape')=='text'
        if not text:ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=5',facecolor=st['fillColor'],edgecolor=st['strokeColor'],lw=.8,linestyle='--' if st.get('dashed')=='1' else '-',zorder=2))
        ax.text(x if text else x+w/2,y+h/2,cell.get('value',''),ha='left' if text else 'center',va='center',fontsize=float(st.get('fontSize',12.5))*.72,fontfamily=st.get('fontFamily','Times New Roman'),fontweight='bold' if st.get('fontStyle')=='1' else 'normal',color=st.get('fontColor','#172B3A'),zorder=3)
    for cell in tree.findall('.//mxCell[@edge="1"]'):
        g=cell.find('mxGeometry');points=[]
        for p in [g.find("mxPoint[@as='sourcePoint']"),*g.findall('Array/mxPoint'),g.find("mxPoint[@as='targetPoint']")]:points.append((float(p.get('x')),float(p.get('y'))))
        dashed='dashed=1' in cell.get('style','');ls='--' if dashed else '-'
        for a,b in zip(points[:-2],points[1:-1]):ax.plot([a[0],b[0]],[a[1],b[1]],color='#202020',lw=1.15,ls=ls,zorder=4)
        ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=11,lw=1.15,linestyle=ls,color='#202020',shrinkA=0,shrinkB=0,zorder=4))
    plt.rcParams['pdf.fonttype']=42
    fig.savefig(ROOT/'figures'/output,metadata={'CreationDate':None,'ModDate':None});plt.close(fig)

def architecture():
    defaults();export(SOURCE/'architecture.drawio','fig_arch.pdf')
def protocol():
    defaults();export(SOURCE/'evaluation_protocol.drawio','evaluation_protocol.pdf')
if __name__=='__main__':architecture();protocol()
