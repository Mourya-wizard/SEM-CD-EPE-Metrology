"""Gauges on the design raster, truth evaluation, and GDS contour export.

Conventions: design-crop px (1 nm), points [x, y], normals [nx, ny] pointing OUT of the feature.
Every gauge is built from edge PROBES: (anchor on the design edge, outward normal). A method measures a signed
offset t (px, + outward) per probe; gauge values are:
  epe     : t                                 (kinds epe1d = straight segments, epe2d = near corners/line ends)
  cd      : design_cd + t_a + t_b             (two opposite probes on a straight segment, corner-excluded)
  contact : 2*sqrt(area/pi) of the polygon of ray probes (anchor at the design boundary along each ray)
"""
import numpy as np
import pandas as pd
from scipy import ndimage

CONTACT_MIN_VALID_FRAC = 0.75   # a contact gauge is valid if >= 75% of its rays are valid


def design_edge_points(T, corner_excl_px, stride=1, border_px=0):
    """Edge points on straight design segments.

    T: (H,W) bool design raster (True = feature). A boundary between pixel (i,j) and (i,j+1) gives a point at
    x = j + 0.5, y = i with outward normal (+1,0) if T[i,j] else (-1,0); same for horizontal boundaries.
    A point is kept only if the same boundary type continues for >= corner_excl_px on both sides along the edge,
    i.e. it is at least corner_excl_px from any corner or jog. Returns pts (N,2), normals (N,2), seg_dir (N,) 0=vertical edge.
    """
    T = T.astype(bool)
    H, W = T.shape
    c = int(corner_excl_px)
    out_p, out_n, out_d = [], [], []
    for sign in (1, -1):                                   # vertical edges: between columns j and j+1
        bv = (T[:, :-1] & ~T[:, 1:]) if sign == 1 else (~T[:, :-1] & T[:, 1:])
        run = ndimage.uniform_filter1d(bv.astype(np.float32), 2 * c + 1, axis=0, mode="constant") > 1 - 1e-6
        ii, jj = np.nonzero(run)
        keep = (ii % stride == 0)
        out_p.append(np.stack([jj[keep] + 0.5, ii[keep].astype(float)], 1))
        out_n.append(np.tile([float(sign), 0.0], (keep.sum(), 1)))
        out_d.append(np.zeros(keep.sum(), int))
    for sign in (1, -1):                                   # horizontal edges: between rows i and i+1
        bh = (T[:-1, :] & ~T[1:, :]) if sign == 1 else (~T[:-1, :] & T[1:, :])
        run = ndimage.uniform_filter1d(bh.astype(np.float32), 2 * c + 1, axis=1, mode="constant") > 1 - 1e-6
        ii, jj = np.nonzero(run)
        keep = (jj % stride == 0)
        out_p.append(np.stack([jj[keep].astype(float), ii[keep] + 0.5], 1))
        out_n.append(np.tile([0.0, float(sign)], (keep.sum(), 1)))
        out_d.append(np.ones(keep.sum(), int))
    pts, nrm, sd = np.concatenate(out_p), np.concatenate(out_n), np.concatenate(out_d)
    if border_px:
        ok = (pts[:, 0] >= border_px) & (pts[:, 0] <= W - 1 - border_px) & \
             (pts[:, 1] >= border_px) & (pts[:, 1] <= H - 1 - border_px)
        pts, nrm, sd = pts[ok], nrm[ok], sd[ok]
    return pts, nrm, sd


def _key(p, n):
    return np.round(p * 2).astype(np.int64)[:, 0] * 10 ** 7 + np.round(p * 2).astype(np.int64)[:, 1] * 10 + \
        (np.round(n[:, 0]).astype(int) + 1) * 3 + (np.round(n[:, 1]).astype(int) + 1)


def build_gauges(T, gc, is_contact, border_px):
    """Build probes and gauges for one design crop. gc = cfg['gauges'].
    Returns DataFrame `probes` with columns gauge_id, kind, x, y, nx, ny, design_cd (cd/contact only, else nan),
    cx, cy (contact centre, else nan)."""
    T = np.asarray(T, bool)
    rows = []
    gid = 0
    # EPE probes: 1D (corner-excluded straight segments) and 2D (remaining boundary points)
    p1, n1, _ = design_edge_points(T, gc["corner_excl_px"], 1, border_px)
    pa, na, _ = design_edge_points(T, 0, 1, border_px)
    k1 = set(_key(p1, n1).tolist())
    is1 = np.array([k in k1 for k in _key(pa, na).tolist()], bool)
    s = gc["epe_stride_px"]
    sel1 = np.nonzero(is1)[0][::s]
    sel2 = np.nonzero(~is1)[0][::s]
    for kind, sel in (("epe1d", sel1), ("epe2d", sel2)):
        for i in sel:
            rows.append((gid, kind, pa[i, 0], pa[i, 1], na[i, 0], na[i, 1], np.nan, np.nan, np.nan)); gid += 1
    if is_contact:
        lab, n = ndimage.label(T)
        H, W = T.shape
        K = gc["contact_rays"]
        ang = 2 * np.pi * np.arange(K) / K
        dirs = np.stack([np.cos(ang), np.sin(ang)], 1)
        for k in range(1, n + 1):
            m = lab == k
            ys, xs = np.nonzero(m)
            if ys.min() < border_px or xs.min() < border_px or ys.max() > H - 1 - border_px or xs.max() > W - 1 - border_px:
                continue
            cy, cx = ys.mean(), xs.mean()
            # design boundary along each ray (bilinear interp of the raster at level 0.5)
            ts = np.arange(0, gc["contact_max_radius_px"], 0.05)
            px = cx + dirs[:, :1] * ts[None]; py = cy + dirs[:, 1:] * ts[None]
            v = ndimage.map_coordinates(m.astype(float), [py.ravel(), px.ravel()], order=1).reshape(px.shape) - 0.5
            first = np.argmax(v < 0, 1)
            if (v[np.arange(K), first] >= 0).any():
                continue
            r = ts[first - 1] + 0.05 * v[np.arange(K), first - 1] / (v[np.arange(K), first - 1] - v[np.arange(K), first])
            dcd = 2 * np.sqrt(m.sum() / np.pi)      # area-equivalent design CD
            r = np.full(K, dcd / 2)                 # probes anchored on the area-equivalent circle: square
            for j in range(K):                      # corners would put diagonal anchors >16 px from round prints
                rows.append((gid, "contact", cx + dirs[j, 0] * r[j], cy + dirs[j, 1] * r[j], dirs[j, 0], dirs[j, 1], dcd, cx, cy))
            gid += 1
    else:
        # CD gauges: pair a +normal 1D probe with the opposite -normal 1D probe across the same feature
        pts, nrm = p1, n1
        keyset = set(_key(pts, nrm).tolist())
        H, W = T.shape
        cs = gc["cd_stride_px"]
        for i in range(len(pts)):
            nx, ny = nrm[i]
            if nx + ny < 0:                     # start only from +x / +y probes
                continue
            x, y = pts[i]
            along = y if nx != 0 else x
            if int(round(along)) % cs:
                continue
            # walk inward (against the normal) through the feature to the opposite boundary
            if nx > 0:
                j = int(x - 0.5); row = int(y)
                jj = j
                while jj >= 0 and T[row, jj]:
                    jj -= 1
                q, qn = np.array([jj + 0.5, y]), np.array([-1.0, 0.0])
                cd = x - (jj + 0.5)
            else:
                i0 = int(y - 0.5); col = int(x)
                ii = i0
                while ii >= 0 and T[ii, col]:
                    ii -= 1
                q, qn = np.array([x, ii + 0.5]), np.array([0.0, -1.0])
                cd = y - (ii + 0.5)
            if _key(q[None], qn[None])[0] not in keyset or cd > gc["cd_max_px"]:
                continue
            rows.append((gid, "cd", x, y, nx, ny, cd, np.nan, np.nan))
            rows.append((gid, "cd", q[0], q[1], qn[0], qn[1], cd, np.nan, np.nan))
            gid += 1
    return pd.DataFrame(rows, columns=["gauge_id", "kind", "x", "y", "nx", "ny", "design_cd", "cx", "cy"])


def gauge_values(probes, t, valid):
    """Combine per-probe offsets t (px, + outward) into gauge values.
    Returns DataFrame gauge_id, kind, value (nm), valid. epe value = t; cd = design_cd + sum t; contact = area-eq CD."""
    df = probes.assign(t=t, v=valid)
    out = []
    for kind, g in df.groupby("kind", sort=False):
        if kind in ("epe1d", "epe2d"):
            out.append(pd.DataFrame({"gauge_id": g.gauge_id.values, "kind": kind, "value": g.t.values, "valid": g.v.values}))
        elif kind == "cd":
            a = g.groupby("gauge_id").agg(design_cd=("design_cd", "first"), ts=("t", "sum"), ok=("v", "all"))
            out.append(pd.DataFrame({"gauge_id": a.index.values, "kind": kind,
                                     "value": a.design_cd.values + a.ts.values, "valid": a.ok.values}))
        else:
            vals, oks, ids = [], [], []
            for gid_, c in g.groupby("gauge_id"):
                K = len(c)
                ok = c.v.values & np.isfinite(c.t.values)
                r = np.hypot(c.x.values - c.cx.values, c.y.values - c.cy.values) + np.where(ok, c.t.values, 0)
                ang = np.arctan2(c.ny.values, c.nx.values)
                if ok.sum() >= 3:
                    a_, r_ = ang[ok], r[ok]
                    o_ = np.argsort(a_); a_, r_ = a_[o_], r_[o_]
                    da = np.diff(np.append(a_, a_[0] + 2 * np.pi))
                    # polygon over valid rays, normalised so a circle of radius r gives exactly CD = 2r
                    area = 0.5 * np.sum(r_ * np.roll(r_, -1) * np.sin(da)) / (K / (2 * np.pi) * np.sin(2 * np.pi / K))
                else:
                    area = 0.0
                vals.append(2 * np.sqrt(max(area, 0) / np.pi))
                oks.append(bool(ok.mean() >= CONTACT_MIN_VALID_FRAC) and bool((r[ok] > 0).all())); ids.append(gid_)
            out.append(pd.DataFrame({"gauge_id": ids, "kind": kind, "value": vals, "valid": oks}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["gauge_id", "kind", "value", "valid"])


class GaugeAggregator:
    """Fast, vectorised equivalent of gauge_values for one fixed probes table (built once per tile).
    Output order and values are identical to gauge_values (tested in tests/test_registration.py)."""

    def __init__(self, probes):
        self.p = probes.reset_index(drop=True)
        k = self.p.kind.values
        self.epe = {kind: np.nonzero(k == kind)[0] for kind in ("epe1d", "epe2d")}
        cd = self.p[self.p.kind == "cd"]
        g = cd.groupby("gauge_id", sort=True)
        self.cd_ids = np.array(list(g.groups.keys()))
        self.cd_idx = np.array([v.values for v in g.groups.values()]) if len(cd) else np.zeros((0, 2), int)
        self.cd_design = g.design_cd.first().values if len(cd) else np.zeros(0)
        ct = self.p[self.p.kind == "contact"]
        gc = ct.groupby("gauge_id", sort=True)
        self.ct_ids = np.array(list(gc.groups.keys()))
        if len(ct):
            idx = [v.values for v in gc.groups.values()]
            self.K = len(idx[0])
            assert all(len(i) == self.K for i in idx)
            self.ct_idx = np.array(idx)
            P = self.p
            self.ct_r0 = np.hypot(P.x.values[self.ct_idx] - P.cx.values[self.ct_idx], P.y.values[self.ct_idx] - P.cy.values[self.ct_idx])
            self.ct_ang = np.arctan2(P.ny.values[self.ct_idx], P.nx.values[self.ct_idx])
        self.order = [kk for kk in dict.fromkeys(k)]          # kinds in first-appearance order (like groupby sort=False)

    def values(self, t, valid):
        t = np.asarray(t, float); valid = np.asarray(valid, bool)
        parts = {}
        for kind, ix in self.epe.items():
            if len(ix):
                parts[kind] = (self.p.gauge_id.values[ix], t[ix], valid[ix])
        if len(self.cd_ids):
            parts["cd"] = (self.cd_ids, self.cd_design + t[self.cd_idx].sum(1), valid[self.cd_idx].all(1))
        if len(self.ct_ids):
            K = self.K
            ok = valid[self.ct_idx] & np.isfinite(t[self.ct_idx])
            r = self.ct_r0 + np.where(ok, t[self.ct_idx], 0)
            norm = K / (2 * np.pi) * np.sin(2 * np.pi / K)
            vals = np.zeros(len(self.ct_ids)); oks = np.zeros(len(self.ct_ids), bool)
            for i in range(len(self.ct_ids)):
                o = ok[i]
                if o.sum() >= 3:
                    a_, r_ = self.ct_ang[i][o], r[i][o]
                    s_ = np.argsort(a_); a_, r_ = a_[s_], r_[s_]
                    da = np.diff(np.append(a_, a_[0] + 2 * np.pi))
                    area = 0.5 * np.sum(r_ * np.roll(r_, -1) * np.sin(da)) / norm
                else:
                    area = 0.0
                vals[i] = 2 * np.sqrt(max(area, 0) / np.pi)
                oks[i] = bool(o.mean() >= CONTACT_MIN_VALID_FRAC) and bool((r[i][o] > 0).all())
            parts["contact"] = (self.ct_ids, vals, oks)
        out = [pd.DataFrame({"gauge_id": parts[kk][0], "kind": kk, "value": parts[kk][1], "valid": parts[kk][2]})
               for kk in self.order if kk in parts]
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["gauge_id", "kind", "value", "valid"])


def truth_offsets(F, probes, half_window):
    """Truth offset per probe = zero crossing of F (bilinear) nearest the anchor; valid iff exactly one crossing."""
    from sem_sim import crossings_along
    t, nc = crossings_along(F, probes[["x", "y"]].values, probes[["nx", "ny"]].values, half_window)
    return t, (nc == 1) & np.isfinite(t)


# ----------------------------------------------------------------------------------------------- GDS
def contours_from_field(F, level=0.0):
    """Sub-pixel contours of a field (marching squares). Returns list of (N,2) [x,y] arrays in crop px."""
    from skimage import measure
    return [c[:, ::-1].copy() for c in measure.find_contours(np.asarray(F, np.float64), level)]


def export_gds(path, contours_by_layer, origin_xy_nm, cell_name="SEM_CONTOURS"):
    """contours_by_layer: {layer: [ (N,2) crop px arrays ]}. origin_xy_nm = tile coords of crop px (0,0).
    1 px = 1 nm. Library unit 1 um, precision 1e-12 m. Closed contours -> polygons, open -> paths (width 1 nm)."""
    import gdstk
    lib = gdstk.Library(unit=1e-6, precision=1e-12)
    cell = lib.new_cell(cell_name)
    for layer, cs in contours_by_layer.items():
        for c in cs:
            xy = (c + np.asarray(origin_xy_nm)[None]) / 1000.0
            xy[:, 1] *= -1                       # image rows go down; GDS y goes up
            if len(c) > 2 and np.allclose(c[0], c[-1]):
                cell.add(gdstk.Polygon(xy[:-1], layer=int(layer)))
            elif len(c) > 1:
                cell.add(gdstk.FlexPath(xy, 0.001, tolerance=1e-7, layer=int(layer), simple_path=True))  # default tol 10 nm drops vertices
    lib.write_gds(path, max_points=8190)       # default 199 would fracture long contours


def gds_roundtrip_check(path, contours_by_layer, origin_xy_nm, tol_nm=1e-3):
    """Read GDS back and compare every vertex with the source contours (design coordinates, nm)."""
    import gdstk
    lib = gdstk.read_gds(path)
    cell = lib.top_level()[0]
    got = {}
    for p in cell.polygons:
        got.setdefault(p.layer, []).append(p.points)
    for fp in cell.paths:
        got.setdefault(fp.layers[0], []).append(fp.spine())
    maxdiff, n = 0.0, 0
    for layer, cs in contours_by_layer.items():
        exp = []
        for c in cs:
            xy = (c + np.asarray(origin_xy_nm)[None]) / 1000.0
            xy[:, 1] *= -1
            exp.append(xy[:-1] if (len(c) > 2 and np.allclose(c[0], c[-1])) else xy)
        g = got.get(layer, [])
        if len(g) != len([e for e in exp if len(e) > 1]):
            return False, np.inf, n
        gs = sorted(g, key=lambda a: (len(a), float(a[:, 0].sum())))
        es = sorted([e for e in exp if len(e) > 1], key=lambda a: (len(a), float(a[:, 0].sum())))
        for a, b in zip(gs, es):
            if a.shape != b.shape:
                return False, np.inf, n
            maxdiff = max(maxdiff, float(np.abs(a - b).max()) * 1000.0)
            n += len(a)
    return maxdiff <= tol_nm, maxdiff, n
