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
    d.box('title',8,5,600,22,'Forecasting path',text=True,bold=True)
    for key,x,w,label in [('input',8,122,'Past case counts\nTraining-scale\nlog transform'),('encoder',153,126,'Encoder\nStatic / adaptive\nrepresentation'),('readout',302,98,'Read-out\nPer horizon'),('head',423,140,'Selected head\nDirect / residual\nGated / SEIR'),('output',586,122,'Inverse transform\nCase forecasts')]:
        d.box(key,x,36,w,66,label,fill='#E8F0F7' if key!='output' else '#E9F3EE',bold=False)
    for k,a,b in [('e1',130,153),('e2',279,302),('e3',400,423),('e4',563,586)]:d.edge(k,[(a,69),(b,69)])
    d.box('conditions',8,122,500,20,'Conditional inputs',text=True,bold=True)
    d.box('anchor',8,151,185,60,'Last observed cases\nPersistence anchor',fill='#F4F1E7',stroke='#726447')
    d.box('seir',229,151,228,60,'SEIR heads only\nPopulation + initial state\nDaily compartment flows',fill='#F4F1E7',stroke='#726447')
    d.edge('anchor_edge',[(193,180),(211,180),(211,114),(466,114),(466,102)])
    d.edge('seir_edge',[(457,180),(519,180),(519,102)])
    d.box('loss',586,151,122,60,'Training only\nTargets + loss\nSquared error / NB',fill='#F4F4F4',stroke='#666666',dashed=True)
    d.edge('loss_edge',[(647,102),(647,151)],True)
    d.box('legend',8,229,700,20,'Solid arrows: inference     Dashed arrow: training only     NB: negative binomial',text=True)
    d.write()
    d=Diagram('evaluation_protocol',350,200)
    d.box('title',5,2,340,21,'Existing period: rolling origins',text=True,bold=True)
    for key,x,w,label,fill in [('train',5,90,'Training','#E8F0F7'),('val',121,92,'Validation','#F4F1E7'),('dev',237,108,'Development\nevaluation','#E9F3EE')]:d.box(key,x,32,w,42,label,fill)
    d.edge('v1',[(95,53),(121,53)]);d.edge('v2',[(213,53),(237,53)])
    d.box('overlap',5,79,340,32,'Forecast starts are partitioned chronologically.\nTarget weeks overlap at boundaries.',text=True)
    d.box('freeze',5,127,124,44,'Select on validation\nFreeze settings',fill='#F4F1E7')
    d.box('confirm',158,127,187,44,'New-period confirmation\nPending; no results',fill='#F4F4F4',stroke='#666666',dashed=True)
    d.edge('future',[(129,149),(158,149)],True)
    d.box('footer',5,177,340,19,'Schematic sequence; spacing does not represent time.',text=True)
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
