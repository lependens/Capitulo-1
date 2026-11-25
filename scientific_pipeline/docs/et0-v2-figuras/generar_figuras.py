"""Reproduce the six published ET0 V2 figures from verified, frozen artifacts."""

import argparse
from collections import Counter
import csv
import gzip
from hashlib import file_digest
from html import escape
import json
from pathlib import Path


def sha(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--retrospective', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, retro, out = args.run, args.retrospective, args.output
    freeze_path = root / 'freeze.json'
    manifest_path = retro / 'manifest.json'
    if sha(freeze_path) != '97acad7ce7cc39c65d82f6efdc36290ed0325caaef2dedb69d9ffbefe1addfbd':
        raise ValueError('Unexpected freeze')
    if sha(manifest_path) != 'f771dc3564270528c203d9af709b2b88649a552ef7dd831bf4503d04a2fd8daa':
        raise ValueError('Unexpected retrospective manifest')
    freeze = json.loads(freeze_path.read_bytes())
    manifest = json.loads(manifest_path.read_bytes())
    for name in ('predictions.csv.gz', 'retrospective-metrics.csv'):
        if sha(retro / name) != manifest['outputs'][name]['sha256']:
            raise ValueError(f'Unexpected hash: {name}')
    stages = {}
    for stage, expected in freeze['stages'].items():
        index = root / stage / 'index.json'
        if sha(index) != expected:
            raise ValueError(f'Unexpected index hash: {stage}')
        cells = {}
        for name, expected_cell in json.loads(index.read_bytes())['cells'].items():
            path = root / stage / name / 'frozen.json'
            if sha(path) != expected_cell:
                raise ValueError(f'Unexpected frozen cell hash: {stage}/{name}')
            cells[name] = json.loads(path.read_bytes())
        stages[stage] = cells
    out.mkdir()

    def write(name, contents):
        (out / name).write_text(contents.rstrip() + '\n', encoding='utf-8', newline='\n')

    def f(value, places=4):
        return f'{float(value):.{places}f}'.replace('.', ',') if value not in ('',None) else '—'

    with (retro/'retrospective-metrics.csv').open(encoding='utf-8', newline='') as stream:
        metrics = {(r['group'],r['key'],r['family'],r['calendar'],r['method']):r for r in csv.DictReader(stream)}

    with gzip.open(retro/'predictions.csv.gz', 'rt', encoding='utf-8', newline='') as stream:
        samples = [r for r in csv.DictReader(stream) if r['station_code']=='IB01' and r['family']=='HG_RS_V1']

    families = {'HS_RA_V1':'HS-Ra','HG_RS_V1':'HG-Rs','VALIANTZAS_V1':'Valiantzas'}
    calendars = {'annual':'Anual','seasonal':'Estacional','monthly':'Mensual'}
    core = ['FORMULA','AHC','ML','ANN_A','ANN_E']

    algs = ['LINEAR','RIDGE','ELASTIC_NET','TREE','RANDOM_FOREST','EXTRA_TREES','HIST_GRADIENT_BOOSTING']
    arch = ['16','32','64','64-32','64-32-16']
    stations = sorted({c['key'][0] for c in stages['AHC'].values()})

    def get(family, cal, method, station=None):
        return metrics[('station' if station else 'global',station or 'all',family,cal,method)]

    def chosen(stage, cell):
        info=stages[stage][cell]
        return next(x for x in info['selected'].values() if x['candidate_id']==info['principal']['candidate_id'])

    def freq(stage, predicate):
        return Counter((x['algorithm'] if stage=='ML' else '-'.join(map(str,x['architecture'])))
                       for key,c in stages[stage].items() if predicate(c['key']) for x in [chosen(stage,key)])

    def svg(name, title, alt, lines, width=850, height=400):
        body = '\n'.join(lines)
        write(name, f'<svg xmlns="http://www.w3.org/2000/svg" role="img" viewBox="0 0 {width} {height}">\n'
              f'<title>{escape(title)}</title><desc>{escape(alt)}</desc>\n'
              f'<rect width="{width}" height="{height}" fill="white"/>\n'
              f'<g font-family="system-ui,sans-serif" fill="#172b41" font-size="14">{body}</g></svg>')

    def label(x,y,s,size=14,anchor='start',fill='#172b41'):
        return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{fill}">{escape(str(s))}</text>'

    palette = ['#007f79','#c17710','#395eaa','#934784','#63767c']

    ahc = {a:stages['AHC'][f'IB01__HG_RS_V1__{a}']['ahc'] for a in calendars}

    lines=[label(28,34,'IB01 · HG-Rs · Coeficientes AHC (DEVELOPMENT)',22),
           label(28,58,'Coeficiente adimensional. Decisiones ajustadas hasta 2019.')]
    minimum, maximum=.82,1
    for val in (.82,.86,.90,.94,.98):
        y=315-(val-minimum)*1200
        lines.extend([f'<path d="M65 {y} H817" stroke="#d9e1ea"/>',label(54,y+5,f(val,2),12,'end')])
    for j, month in enumerate(range(1,13)):
        x=93+j*62
        season=('DJF','MAM','JJA','SON')[(month%12)//3]
        for off,g,color in [(-9,ahc['monthly']['groups'][f'{month:02}']['coefficient'],palette[0]),
                            (9,ahc['seasonal']['groups'][season]['coefficient'],palette[1])]:
            y=315-(g-minimum)*1200
            lines.append(f'<circle cx="{x+off}" cy="{y}" r="5" fill="{color}"/>')
        lines.append(label(x,340,f'{month:02}',12,'middle'))
    y=315-(ahc['annual']['groups']['annual']['coefficient']-minimum)*1200
    lines.append(f'<path d="M65 {y} H817" stroke="{palette[2]}" stroke-dasharray="6 4"/>')
    for j,(name,col) in enumerate(zip(['Mensual','Estacional','Anual (línea)'],palette)):
        lines.append(label(100+225*j,375,name,15,'start',col))
    svg('ahc-ib01.svg','Coeficientes AHC IB01', 'Doce coeficientes mensuales y cuatro estacionales proyectados en sus meses.',lines)

    valid_stations=[st for st in stations if st not in ('IB07','IB09','IB101')]
    heat_rows=[[get('HG_RS_V1','annual',m,s) for m in core] for s in valid_stations]
    lines=[label(28,34,'TEST 2020–2025 · HG-Rs anual · RMSE por estación',22),
           label(28,59,'mm/día respecto a PM calculado · el número de pares cambia por estación')]
    for j,method in enumerate(core):
        lines.append(label(175+129*j,91,method,14,'middle'))
    for i,(st,records) in enumerate(zip(valid_stations,heat_rows)):
        y=101+30*i
        assert len({r['common_keys_sha256'] for r in records})==1
        lines.append(label(60,y+20,st,13,'middle'))
        for j,r in enumerate(records):
            v=float(r['rmse'])
            t=max(0,min(1,(v-.28)/.47))
            color=f'rgb({int(224-164*t)},{int(242-128*t)},{int(245-88*t)})'
            lines.append(f'<rect x="{116+129*j}" y="{y}" width="118" height="26" rx="3" fill="{color}"/>')
            lines.append(label(175+129*j,y+19,f(v,3),13,'middle','#fff' if t>.60 else '#172b41'))
    lines.append(label(28,397,'Escala de color fija · cifras en cada celda · fuente: retrospective-metrics.csv'))
    svg('rmse-estaciones.svg','RMSE retrospectivo por estación', 'Valores de cinco modelos en nueve estaciones en HG-Rs anual.',lines,height=419)

    lines=[label(28,35,'TEST 2020–2025 · RMSE según familia y calendario',22),
           label(28,58,'mm/día respecto a PM calculado · 19.251 pares por comparación')]
    for tick in (0,.25,.5,.75,1):
        y=309-220*tick
        lines.extend([f'<path d="M59 {y} H823" stroke="#d9e1ea"/>',label(50,y+5,f(tick,2),12,'end')])
    for i,fam in enumerate(families):
        cx=73+251*i
        for k,cal in enumerate(calendars):
            for j,m in enumerate(core):
                v=float(get(fam,cal,m)['rmse'])
                x=cx+79*k+12*j
                lines.append(f'<rect x="{x}" y="{309-220*v}" width="10" height="{220*v}" fill="{palette[j]}"/>')
            lines.append(label(cx+79*k+29,329,['Anual','Estac.','Mensual'][k],11,'middle'))
        lines.append(label(cx+108,356,families[fam],15,'middle'))
    for j,m in enumerate(core):
        lines.append(label(35+156*j,389,m,13,'start',palette[j]))
    svg('rmse-global.svg','RMSE según familia y calendario', 'Cinco métodos por familia y segmentación.',lines,height=410)

    lines=[label(28,34,'DEVELOPMENT · Configuraciones seleccionadas',22),
           label(28,59,'Frecuencias de selección por celda, sin ordenar por TEST')]
    for stage,x0,maxn,values in [('ML',32,105,algs),('ANN_A',520,105,arch)]:
        lines.append(label(x0,92,'ML principal' if stage=='ML' else 'Arquitecturas ANN A / E',17))
        for i,it in enumerate(values):
            y=111+35*i
            lines.append(label(x0,y+13,it,11))
            for j,s in enumerate([stage] if stage=='ML' else ['ANN_A','ANN_E']):
                v=freq(s,lambda k:True)[it]
                off=40 if stage!='ML' else 186
                width=v*2.45
                lines.append(f'<rect x="{x0+off}" y="{y+13*j}" width="{width}" height="11" fill="{palette[j]}"/>')
                lines.append(label(x0+off+width+5,y+9+13*j,v,11))
    lines.extend([label(520,330,'ANN A',13,'start',palette[0]),label(620,330,'ANN E',13,'start',palette[1])])
    svg('seleccion-development.svg','Frecuencias de selección', 'Siete métodos principales ML y arquitecturas por regla A y E.',lines,height=381)

    series = sorted([r for r in samples if r['calendar']=='annual' and r['date'][:4]=='2023' and
                     all(r[k] for k in ['target',*core])],key=lambda r:r['date'])
    assert len(series)==365
    lines=[label(28,35,'IB01 · HG-Rs anual · TEST 2023',22),
           label(28,58,'Serie diaria · mm/día · referencia y tres estimaciones')]
    for tick in (0,2,4,6,8,10):
        y=301-21*tick
        lines.extend([f'<path d="M60 {y} H818" stroke="#e2e8ef"/>',label(50,y+4,tick,12,'end')])
    for key,color in [('target','#172b41'),('FORMULA',palette[0]),('AHC',palette[1]),('ANN_A',palette[3])]:
        pts=' '.join(f'{60+i*2.05:.2f},{301-21*float(r[key]):.2f}' for i,r in enumerate(series))
        lines.append(f'<polyline points="{pts}" stroke="{color}" fill="none" stroke-width="1.25"/>')
    for j,(k,name,col) in enumerate([('target','PM calculado','#172b41'),('FORMULA','Fórmula',palette[0]),
                                       ('AHC','AHC',palette[1]),('ANN_A','ANN A',palette[3])]):
        lines.append(label(70+190*j,372,name,14,'start',col))
    for i,r in enumerate(series):
        if r['date'][8:]=='01' and int(r['date'][5:7])%2:
            lines.append(label(60+i*2.05,327,r['date'][5:7],12,'middle'))
    svg('ib01-serie-test.svg','Serie diaria HG-Rs en IB01, 2023','Días de 2023 con PM y cinco métodos principales disponibles.',lines,height=396)

    rows = [r for r in samples if r['calendar']=='annual' and all(r[k] for k in ['target',*core])]
    assert len(rows)==2192
    lines=[label(28,35,'IB01 · HG-Rs anual · Dispersión y residuos TEST',22),
           label(28,58,'ANN A · 2.192 pares diarios · mm/día respecto a PM calculado'),
           label(70,87,'Dispersión de estimaciones (todas las fechas)',14),
           label(475,87,'Histograma del residuo (estimación − PM)',14)]
    largest=max(max(float(r['target']),float(r['ANN_A'])) for r in rows)
    lim=int(largest)+1
    for tick in range(0,lim+1,2):
        x=63+tick*285/lim
        y=331-tick*225/lim
        lines.extend([f'<path d="M{x} 105 V331 M63 {y} H348" stroke="#e2e8ef"/>',
                      label(x,351,tick,11,'middle'),label(54,y+4,tick,11,'end')])
    lines.append('<path d="M63 331 L348 106" stroke="#172b41" stroke-dasharray="5 4"/>')
    for r in rows:
        x=63+285/lim*float(r['target'])
        y=331-225/lim*float(r['ANN_A'])
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.25" fill="#007f79" opacity=".23"/>')
    res=[float(r['ANN_A'])-float(r['target']) for r in rows]
    div=.25
    lo=int(min(res)/div)-1
    hi=int(max(res)/div)+1
    freqs=Counter(int(v//div) for v in res)
    for tick in range(lo,hi+1):
        x=466+(tick-lo)*335/(hi-lo+1)
        n=freqs[tick]
        h=n*202/max(freqs.values())
        lines.append(f'<rect x="{x:.1f}" y="{331-h:.1f}" width="{335/(hi-lo+1)-1:.1f}" height="{h:.1f}" fill="#395eaa"/>')
        if tick%4==0:
            lines.append(label(x,351,tick*div,11,'middle'))
    lines.extend([label(205,378,'PM calculado (eje vertical: ANN A)',13,'middle'),
                  label(628,378,f'Residuo · intervalos de 0,25 mm/día · máximo {max(freqs.values())} pares',12,'middle')])
    svg('ib01-residuos-test.svg','Dispersión y residuos de ANN A en IB01','Cada fecha aparece una vez por panel. La recta discontinua es la identidad.',lines,height=398)


if __name__ == '__main__':
    main()
