"""결과 XML 점검: 컷 연속성, 앵글 영상 범위, 연속 앵글, 타이틀 겹침."""
import sys, xml.etree.ElementTree as ET
from fractions import Fraction as F
t = lambda s: F(s.rstrip('s'))

def verify(path):
    r = ET.parse(path).getroot(); cov = {}
    for a in r.iter('mc-angle'):
        cov[a.get('angleID')] = [(t(x.get('offset')), t(x.get('offset')) + t(x.get('duration'))) for x in a if x.tag == 'asset-clip']
    cl = list(r.iter('mc-clip')); bad = []
    for i, c in enumerate(cl):
        if i and t(c.get('offset')) != t(cl[i-1].get('offset')) + t(cl[i-1].get('duration')): bad.append(('gap', i))
        v = c.find('mc-source[@srcEnable="video"]').get('angleID'); s0 = t(c.get('start')); s1 = s0 + t(c.get('duration'))
        if not any(a <= s0 and s1 <= b for a, b in cov[v]): bad.append(('coverage', i))
        if i and v == cl[i-1].find('mc-source[@srcEnable="video"]').get('angleID'): bad.append(('repeat', i))
    ti = list(r.iter('title'))
    ov = sum(1 for i in range(len(ti) - 1) if t(ti[i+1].get('offset')) < t(ti[i].get('offset')) + t(ti[i].get('duration')))
    return {'clips': len(cl), 'titles': len(ti), 'title_overlaps': ov, 'problems': bad}

if __name__ == '__main__':
    print(verify(sys.argv[1]))
