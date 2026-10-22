#!/usr/bin/env python3
"""
Collage layout optimiser for THIS KID BREATHES MUSIC reel.

Canvas: 1080 x 1920.  Produces layout.json with, for every poster (in arrival
order): centre (cx, cy), displayed square size, rotation (deg, counter-clockwise).

Constraints / goals (all from the brief):
  * title-safe area (Y < 385) stays clear
  * dome core stays largely unobstructed (covers only flank it, or touch the rim)
  * the person's head / arm / upper body stay visible
  * every cover fully inside the canvas (14 px margin)
  * controlled overlaps (~10-30 %), every cover keeps >= ~58 % of its artwork visible
  * spread across left / middle / right and upper / lower, no grid alignment
  * consecutive arrivals are spread apart so the build reads as a scattered, rhythmic stack
Deterministic: fixed RNG seed.
"""
import json, math, sys
import numpy as np
import cv2

W, H = 1080, 1920
N = 28
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 7
rng = np.random.default_rng(SEED)

# ------------------------------------------------------------------ sizes / rotations
# 2 occasional larger, 7 main, 19 smaller (px, displayed square side)
large = [292, 284]
main = [256, 250, 246, 242, 238, 234, 230]
small = [205, 202, 199, 196, 193, 190, 187, 184, 181, 178, 175, 200, 194, 188, 182, 176, 198, 186, 180]
sizes = np.array(large + main + small, dtype=float)
# interleave sizes through the arrival sequence so the build has varied rhythm
order = rng.permutation(N)
sizes = sizes[order]
# keep first poster mid-sized and avoid two larges back-to-back
rot = np.concatenate([rng.uniform(-4.2, 4.2, N - 4), [5.6, -5.4, 6.2, -6.0]])
rot = rot[rng.permutation(N)]
rot[np.abs(rot) < 0.6] = rng.choice([-1.4, 1.1, -0.9, 1.7], size=int((np.abs(rot) < 0.6).sum()))

# ------------------------------------------------------------------ forbidden zones (x0,y0,x1,y1)
ZONES = [
    (0, 0, W, 375),            # title-safe region
    (332, 300, 652, 445),      # dome crown + finial (stepped approximation of the dome outline)
    (268, 445, 738, 525),      # dome shoulders
    (226, 525, 776, 618),      # dome base (lower rim of the dome may be touched below y=618)
    (742, 1030, W, 1300),      # person: head + raised arm
    (684, 1290, W, H),         # person: shoulders / torso
]
MARGIN = 14
BOTTOM_LIMIT = 1500

SC = 4                          # raster scale for visibility estimation
RW, RH = W // SC, H // SC

import glob, re, os
_files = sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                       'assets', 'posters_raw', '*.jpg')),
                key=lambda f: 0 if f.endswith('unnamed.jpg') else int(re.search(r'\((\d+)\)', f).group(1)))


def _saliency(path):
    """Gradient-energy map: where the artwork has text / faces / detail (sums to 1)."""
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1.2)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    m = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), 9)
    m = m + 0.25 * m.mean()                       # floor, so flat areas still count a bit
    return m / m.sum()


_SAL0 = [_saliency(f) for f in _files]
SAL = None


def build_sal_sprites():
    """Rotated, rescaled saliency sprites at raster resolution (rotation is fixed)."""
    global SAL
    SAL = []
    for i in range(N):
        b = max(4, int(round(sizes[i] / SC)))
        base = cv2.resize(_SAL0[i], (b, b), interpolation=cv2.INTER_AREA)
        side = int(b * 1.16) + 4
        M = cv2.getRotationMatrix2D(((b - 1) / 2, (b - 1) / 2), rot[i], 1.0)
        M[0, 2] += (side - b) / 2; M[1, 2] += (side - b) / 2
        spr = cv2.warpAffine(base, M, (side, side), flags=cv2.INTER_LINEAR, borderValue=0)
        spr /= max(spr.sum(), 1e-9)
        SAL.append((spr, side))


def sal_visibility(lab, cx, cy):
    out = np.zeros(N)
    for i in range(N):
        spr, side = SAL[i]
        ox = int(round(cx[i] / SC - side / 2)); oy = int(round(cy[i] / SC - side / 2))
        xa, ya = max(0, ox), max(0, oy); xb, yb = min(RW, ox + side), min(RH, oy + side)
        if xa >= xb or ya >= yb:
            continue
        sub = lab[ya:yb, xa:xb] == (i + 1)
        out[i] = (spr[ya - oy:yb - oy, xa - ox:xb - ox] * sub).sum()
    return out


def corners(cx, cy, s, a):
    th = math.radians(a)
    c, sn = math.cos(th), math.sin(th)
    h = s / 2
    pts = []
    for x, y in ((-h, -h), (h, -h), (h, h), (-h, h)):
        pts.append((cx + x * c + y * sn, cy - x * sn + y * c))
    return np.array(pts)


def bbox_half(s, a):
    th = math.radians(a)
    return s / 2 * (abs(math.cos(th)) + abs(math.sin(th)))


def energy(P, detail=False):
    cx, cy, ang = P[:, 0], P[:, 1], P[:, 2]
    # ---- raster visibility
    lab = np.zeros((RH, RW), np.int32)
    areas = np.zeros(N)
    for i in range(N):
        poly = (corners(cx[i], cy[i], sizes[i], ang[i]) / SC).astype(np.int32)
        cv2.fillPoly(lab, [poly], i + 1)
        areas[i] = (sizes[i] / SC) ** 2
    cnt = np.bincount(lab.ravel(), minlength=N + 1)[1:]
    vis = cnt / areas
    E = 0.0
    sv = sal_visibility(lab, cx, cy)
    E += 60.0 * np.sum(np.maximum(0, 0.62 - vis) ** 2)
    E += 160.0 * np.sum(np.maximum(0, 0.74 - sv) ** 2)       # protect text / faces / detail
    E += 5.0 * np.sum((1 - vis) ** 2) + 6.0 * np.sum((1 - sv) ** 2)

    # ---- pair overlaps (axis aligned approximations)
    hs = np.array([bbox_half(sizes[i], ang[i]) for i in range(N)])
    x0, x1, y0, y1 = cx - hs, cx + hs, cy - hs, cy + hs
    iu = np.triu_indices(N, 1)
    ox = (np.minimum(x1[:, None], x1[None, :]) - np.maximum(x0[:, None], x0[None, :]))
    oy = (np.minimum(y1[:, None], y1[None, :]) - np.maximum(y0[:, None], y0[None, :]))
    A = (2 * hs) ** 2
    minA = np.minimum(A[:, None], A[None, :])
    ov = (ox > 0) & (oy > 0)
    O = np.where(ov, ox * oy / minA, 0.0)
    np.fill_diagonal(O, 0.0)
    maxo = O.max(axis=1)
    Ou = O[iu]
    E += 28.0 * np.sum(np.maximum(0, Ou - 0.30) ** 2)
    DX = np.abs(cx[:, None] - cx[None, :]); DY = np.abs(cy[:, None] - cy[None, :])
    D = np.hypot(DX, DY)
    ms = np.minimum(sizes[:, None], sizes[None, :]) * 0.38
    near = (D < ms)[iu] & ov[iu]
    E += 20.0 * np.sum(np.where(near, (1 - D[iu] / ms[iu]) ** 2, 0.0))
    dxu, dyu = DX[iu], DY[iu]
    E += 1.2 * np.sum(np.where((dxu < 12) & (dyu < 420), (1 - dxu / 12) ** 2, 0.0))
    E += 1.2 * np.sum(np.where((dyu < 12) & (dxu < 420), (1 - dyu / 12) ** 2, 0.0))
    E += 10.0 * np.sum(np.maximum(0, 0.07 - maxo) ** 2)

    # ---- hard-ish boundaries and zones
    for i in range(N):
        pen = 0.0
        pen += max(0, MARGIN - x0[i]) + max(0, x1[i] - (W - MARGIN))
        pen += max(0, 375 - y0[i]) + max(0, y1[i] - BOTTOM_LIMIT)
        E += 4.0 * pen
        for (zx0, zy0, zx1, zy1) in ZONES[1:]:
            ox = min(x1[i], zx1) - max(x0[i], zx0)
            oy = min(y1[i], zy1) - max(y0[i], zy0)
            if ox > 0 and oy > 0:
                E += 0.35 * (ox + oy) * 1.0 + 120.0 * (ox * oy) / (sizes[i] ** 2)

    # ---- distribution: left / middle / right and upper / lower
    left = np.sum(cx < 360); mid = np.sum((cx >= 360) & (cx < 720)); right = np.sum(cx >= 720)
    E += 4.0 * (max(0, 8 - left) ** 2 + max(0, 8 - mid) ** 2 + max(0, 6 - right) ** 2)
    upper = np.sum(cy < 760); lower = np.sum(cy > 1250)
    E += 4.0 * (max(0, 6 - upper) ** 2 + max(0, 4 - lower) ** 2)
    E += 0.0006 * ((cx.mean() - 470) ** 2 + (cy.mean() - 950) ** 2)

    # ---- arrival rhythm: consecutive covers should not land on top of each other
    for i in range(N - 1):
        d = math.hypot(cx[i] - cx[i + 1], cy[i] - cy[i + 1])
        if d < 240:
            E += 2.5 * (1 - d / 240) ** 2
    if detail:
        return E, vis, maxo, sv
    return E


def init():
    P = np.zeros((N, 3))
    for i in range(N):
        while True:
            x = rng.uniform(120, 960); y = rng.uniform(430, 1440)
            if energy_zone_ok(x, y, sizes[i]):
                break
        P[i] = (x, y, rot[i])
    return P


def energy_zone_ok(x, y, s):
    h = s / 2
    for (zx0, zy0, zx1, zy1) in ZONES:
        if x + h > zx0 and x - h < zx1 and y + h > zy0 and y - h < zy1:
            return False
    return True


def anneal(P, iters=36000):
    E = energy(P)
    best, bestE = P.copy(), E
    T0, T1 = 3.0, 0.03
    for k in range(iters):
        T = T0 * (T1 / T0) ** (k / iters)
        sig = 90 * (1 - k / iters) + 8
        Q = P.copy()
        r = rng.random()
        if r < 0.88:
            i = rng.integers(N)
            Q[i, 0] += rng.normal(0, sig); Q[i, 1] += rng.normal(0, sig)
        else:
            i, j = rng.integers(N, size=2)
            if i == j: continue
            Q[[i, j], 0:2] = Q[[j, i], 0:2]
        En = energy(Q)
        if En < E or rng.random() < math.exp(-(En - E) / T):
            P, E = Q, En
            if E < bestE:
                best, bestE = P.copy(), E
        if k % 6000 == 0:
            print(f"iter {k:6d}  T={T:.3f}  E={E:.2f}  best={bestE:.2f}", flush=True)
    return best, bestE


if __name__ == "__main__":
    build_sal_sprites()
    P = init()
    P, E = anneal(P)
    E, vis, maxo, sv = energy(P, detail=True)
    print(f"final E={E:.2f}  min vis={vis.min():.2f}  mean vis={vis.mean():.2f}  min sal-vis={sv.min():.2f}  mean sal-vis={sv.mean():.2f}")
    out = []
    for i in range(N):
        out.append(dict(order=i, cx=round(float(P[i, 0]), 1), cy=round(float(P[i, 1]), 1),
                        size=int(sizes[i]), angle=round(float(P[i, 2]), 2),
                        visible=round(float(vis[i]), 3), salient_visible=round(float(sv[i]), 3), max_overlap=round(float(maxo[i]), 3)))
    fn = f'layout_seed{SEED}.json'
    json.dump(out, open(fn, 'w'), indent=1)
    print('saved', fn, 'E=%.2f minvis=%.2f meanvis=%.2f minsal=%.2f meansal=%.2f' % (E, vis.min(), vis.mean(), sv.min(), sv.mean()))
