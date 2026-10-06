"""Check saved scientific outputs, bibliography structure and the built PDF.

Writes a reproducible report and render previews. Previews are verification
artifacts, never figures used by the manuscript. Does not train or read new cases.
"""
import hashlib
import json
import re
from pathlib import Path

import fitz
import numpy as np
from review_analysis import holm
from common import IEEE, RES, REPO, P9_FILE, load_json, project_stats as ps


def main():
    tex=(IEEE/'main.tex').read_text(encoding='utf-8')
    bib=(IEEE/'refs.bib').read_text(encoding='utf-8')
    keys=re.findall(r'@\w+\{([^,]+),',bib)
    cited={key.strip() for group in re.findall(r'\\cite\{([^}]+)\}',tex) for key in group.split(',')}
    stats=load_json(RES/'development_statistics.json');rows=load_json(P9_FILE)
    assert stats['sha256']==hashlib.sha256(P9_FILE.read_bytes()).hexdigest()
    for metric in ('RMSE','val_RMSE'):
        values=ps.compare(rows,metric,'persistence','origin')
        actual={'better':sum(r['p_adj']<.05 and r['delta']<0 for r in values),
                'worse':sum(r['p_adj']<.05 and r['delta']>0 for r in values),'family':len(values)}
        assert actual==stats['counts'][metric]
        adjusted=holm([r['p'] for r in values])
        actual_holm={'better':sum(p<.05 and r['delta']<0 for p,r in zip(adjusted,values)),
                     'worse':sum(p<.05 and r['delta']>0 for p,r in zip(adjusted,values)), 'family':len(values)}
        assert actual_holm==stats['holm_counts'][metric]
        if metric=='RMSE':
            for r,p in zip(values,adjusted):
                assert np.isclose(stats['arms'][r['arm']]['p_holm'],p)
                assert np.isclose(stats['arms'][r['arm']]['p_adj'],r['p_adj'])
    for name, s in stats['arms'].items():
        rr=[r for r in rows if r['name']==name]
        assert np.isclose(s['rmse'],np.mean([r['RMSE'] for r in rr]))
        assert np.isclose(s['mae'],np.mean([r['MAE'] for r in rr]))
        assert s['origins']==9
        assert s['rows']==(9 if name=='persistence' else 27)
    report={'source_sha256':stats['sha256'],'historical_bh_counts':stats['counts'],'protocol_holm_counts':stats['holm_counts'],
            'duplicate_bib_keys':sorted({k for k in keys if keys.count(k)>1}),
            'missing_citations':sorted(cited-set(keys)), 'unused_bib_keys':sorted(set(keys)-cited)}
    entries={m.group(1):e for e in re.split(r'\n(?=@)',bib) if (m:=re.match(r'@\w+\{([^,]+),',e))}
    report['cited_without_doi_or_url']=[k for k in sorted(cited) if not re.search(r'\bdoi\s*=|\\url\{|\burl\s*=',entries[k])]
    assert not report['duplicate_bib_keys'] and not report['missing_citations']
    evidence=(REPO/'paper/EVIDENCE.md').read_text(encoding='utf-8')
    tags=set(re.findall(r'EV-\d+',evidence)); report['unknown_evidence_ids']=sorted(set(re.findall(r'EV-\d+',tex))-tags)
    assert not report['unknown_evidence_ids']
    preview=RES/'preview';preview.mkdir(exist_ok=True)
    figs=[]
    supplement=(IEEE/'supplement.tex').read_text(encoding='utf-8')
    names=list(dict.fromkeys(re.findall(r'\\includegraphics\[[^]]*\]\{([^}]+)\}',tex+'\n'+supplement)))
    for name in names:
        d=fitz.open(IEEE/name);page=d[0]
        spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans']]
        figs.append({'path':name,'width_inches':round(page.rect.width/72,4),
                     'height_inches':round(page.rect.height/72,4),'raster_images':len(page.get_images()),
                     'minimum_font_pt':min(s['size'] for s in spans),'maximum_font_pt':max(s['size'] for s in spans)})
        assert len(page.get_images())==0
        assert abs(page.rect.width/72-(7.16 if 'arch' in name else 3.5))<.02
        page.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(preview/(Path(name).stem+'.png'))
        page.get_pixmap(matrix=fitz.Matrix(1.5,1.5),colorspace=fitz.csGRAY).save(preview/(Path(name).stem+'_gray.png'))
    report['figures']=figs
    log=(IEEE/'main.log').read_text(encoding='utf-8',errors='replace')
    report['latex_warnings']=[l for l in log.splitlines() if 'Warning' in l or 'Overfull' in l or 'Underfull' in l]
    assert not any('undefined' in l or 'Overfull' in l for l in report['latex_warnings'])
    pdf=fitz.open(IEEE/'main.pdf')
    report['pages']=len(pdf)
    assert len(pdf)<=6, 'Main exceeds the six-page maximum'
    report['references_start_page']=next((i+1 for i,p in enumerate(pdf) if 'REFERENCES' in p.get_text()),None)
    report['fonts_have_type3']=any('Type3' in f for p in pdf for f in p.get_fonts())
    assert not report['fonts_have_type3']
    for i,page in enumerate(pdf):
        page.get_pixmap(matrix=fitz.Matrix(1,1)).save(preview/f'page_{i+1}.png')
    supplementary_pdf=fitz.open(IEEE/'supplement.pdf')
    report['supplement_pages']=len(supplementary_pdf)
    report['supplement_latex_warnings']=[l for l in (IEEE/'supplement.log').read_text(encoding='utf-8',errors='replace').splitlines() if 'Warning' in l or 'Overfull' in l or 'Underfull' in l]
    assert not any('undefined' in l or 'Overfull' in l for l in report['supplement_latex_warnings'])
    for i,page in enumerate(supplementary_pdf):
        page.get_pixmap(matrix=fitz.Matrix(1,1)).save(preview/f'supplement_page_{i+1}.png')
    abstract=tex.split(r'\begin{abstract}')[1].split(r'\end{abstract}')[0]
    abstract=re.sub(r'(?<!\\)%[^\n]*','',abstract)
    report['abstract_words']=len(abstract.split())
    (RES/'publication_checks.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
