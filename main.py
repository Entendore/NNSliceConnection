"""
GPS-theta v8 — weight-field charts + the ELEVATOR (off-span axis).

Frame: K hard-coded anchors -> SVD -> orthonormal chart E ((K-1) x N), mean.
Elevator: unit n-hat orthogonal to span(E), chosen toward dest/nearest/random.
  theta = mean + c @ E + h * n-hat      (c: K-1 chart sliders, h: elevator)
Navigate now lands EXACTLY on trained nets (h absorbs the old landing error).

Charts are pure WEIGHT fields (closed form, no data): rho_i, Drho^2, ||theta||,
cos, Voronoi distance/cells, per-tensor norms — with optional contour lines.
Accuracy survives only as a demoted optional overlay.

New: barycentric composition bars (theta_hat = sum lambda_i a_i, sum=1),
L/direct/via-anchor route shapes, snapshots, morphology click-to-plot,
field gallery window, constellation geometry study, journal export,
solver-residual readout, arrow-key nudges.
Reuses net_vectors_v4.npz (delete to retrain the 13 destination nets).
"""

import os
os.environ.setdefault("QT_API", "PySide6")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys, time
from threading import Lock

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from PySide6.QtCore import Qt, QThread, Signal, QTimer, QEventLoop, QRectF, QPointF
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QAction, QShortcut, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QComboBox, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QTextEdit, QSpinBox,
    QProgressDialog, QCheckBox, QScrollArea, QSplitter, QDialog,
)

import matplotlib as mpl
mpl.use("qtagg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# ================================================================ config
SEED, MAX_F, MAX_C = 0, 64, 10
HIDDEN, EPOCHS, BATCH, LR = (32, 32), 300, 32, 1e-3
CACHE = "net_vectors_v4.npz"
CORE_SATS = ["const:0.1", "const:0.5", "const:0.9"]
MAX_FLEET = 16
FIELDS = ["range rho_i", "Drho^2 vs SAT1", "||theta||", "cos(theta, a_i)",
          "Voronoi distance", "Voronoi cells", "tensor norm"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================================================================ ML core
def to_class_ids(y):
    y = np.asarray(y, float).ravel()
    if len(np.unique(y)) > MAX_C:
        edges = np.quantile(y, np.linspace(0, 1, MAX_C + 1)[1:-1])
        y = np.searchsorted(edges, y, side="right")
    return np.unique(y, return_inverse=True)[1].astype(np.int64)

def get_data():
    loaders = {"iris": datasets.load_iris, "wine": datasets.load_wine,
               "cancer": datasets.load_breast_cancer, "digits": datasets.load_digits,
               "diab": datasets.load_diabetes, "linn": datasets.load_linnerud}
    out = {}
    for name, loader in loaders.items():
        ds = loader()
        X = np.asarray(ds.data, float)
        y = to_class_ids(ds.target[:, 0] if np.asarray(ds.target).ndim > 1 else ds.target)
        try:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25,
                                                  random_state=SEED, stratify=y)
        except ValueError:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=SEED)
        sc = StandardScaler().fit(Xtr)
        def pad(a, sc=sc):
            o = np.zeros((a.shape[0], MAX_F), np.float32)
            o[:, :a.shape[1]] = sc.transform(a)
            return o
        out[name] = (pad(Xtr), ytr, pad(Xte), yte)
    return out

def make_model():
    layers, d = [], MAX_F
    for h in HIDDEN:
        layers += [nn.Linear(d, h), nn.ReLU()]; d = h
    layers.append(nn.Linear(d, MAX_C))
    return nn.Sequential(*layers).to(DEVICE)

def train(model, X, y):
    Xt, yt = torch.from_numpy(X).to(DEVICE), torch.from_numpy(y).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(EPOCHS):
        perm = torch.randperm(len(Xt), device=DEVICE)
        for i in range(0, len(Xt), BATCH):
            idx = perm[i:i+BATCH]
            opt.zero_grad(); F.cross_entropy(model(Xt[idx]), yt[idx]).backward(); opt.step()
    return model

def flat_of(m):
    return torch.cat([p.data.reshape(-1) for p in m.parameters()]
                     ).cpu().numpy().astype(np.float64)

def set_flat(m, vec):
    i = 0
    with torch.no_grad():
        for p in m.parameters():
            n = p.numel()
            p.copy_(torch.from_numpy(vec[i:i+n].astype(np.float32)).reshape_as(p)); i += n

# -------- synthetic anchor generators (deterministic, training-free) ----------
def sat_const(c):   return lambda N, S: np.full(N, c)
def sat_prng(seed):
    def gen(N, S):
        m, rs = make_model(), np.random.RandomState(seed)
        with torch.no_grad():
            for mod in m.modules():
                if isinstance(mod, nn.Linear):
                    b = 1.0 / np.sqrt(mod.in_features)
                    mod.weight.copy_(torch.from_numpy(
                        rs.uniform(-b, b, tuple(mod.weight.shape)).astype(np.float32)))
                    mod.bias.copy_(torch.from_numpy(
                        rs.uniform(-b, b, tuple(mod.bias.shape)).astype(np.float32)))
        return flat_of(m)
    return gen
def sat_ramp(a, b):   return lambda N, S: np.linspace(a, b, N)
def sat_sin(k, amp):  return lambda N, S: amp*np.sin(2*np.pi*k*np.arange(N)/N)
def sat_sign(k, amp): return lambda N, S: amp*np.sign(np.sin(2*np.pi*k*np.arange(N)/N))
def sat_onehot(t, v):
    def gen(N, S):
        out, i = np.zeros(N), 0
        for j, sh in enumerate(S):
            n = int(np.prod(sh))
            if j == t: out[i:i+n] = v
            i += n
        return out
    return gen

class Core:
    def __init__(self):
        self.lock = Lock()
        self.data = get_data(); self.ds_list = list(self.data)
        self.model = make_model()
        self.Xbig = torch.cat([torch.from_numpy(self.data[d][2]) for d in self.ds_list]).to(DEVICE)
        self.ybig = torch.cat([torch.from_numpy(self.data[d][3]) for d in self.ds_list]).to(DEVICE)
        self.bounds = np.cumsum([0] + [len(self.data[d][2]) for d in self.ds_list])
        self.nets = None

    def load_from_cache(self):
        z = np.load(CACHE); self.nets = {k: z[k] for k in z.files}

    def train_all(self, progress=None):
        nets, tasks = {}, []
        for i, name in enumerate(self.ds_list):
            Xtr, ytr = self.data[name][0], self.data[name][1]
            tasks += [(f"spec:{name}", Xtr, ytr, SEED + i*10),
                      (f"spec2:{name}", Xtr, ytr, SEED + 1000 + i*10)]
        Xa = np.concatenate([self.data[d][0] for d in self.ds_list])
        ya = np.concatenate([self.data[d][1] for d in self.ds_list])
        tasks.append(("gen:ALL", Xa, ya, SEED))
        for k, (name, X, y, sd) in enumerate(tasks):
            if progress: progress(k, len(tasks), name)
            torch.manual_seed(sd)
            nets[name] = flat_of(train(make_model(), X, y))
        self.nets = nets
        np.savez(CACHE, **nets)
        if progress: progress(len(tasks), len(tasks), "done")

    def metrics(self, vec):     # used ONLY by the optional overlay
        with self.lock:
            set_flat(self.model, vec)
            with torch.no_grad():
                logits = self.model(self.Xbig)
            out = {}
            for i, d in enumerate(self.ds_list):
                lg, yy = logits[self.bounds[i]:self.bounds[i+1]], self.ybig[self.bounds[i]:self.bounds[i+1]]
                out[d] = ((lg.argmax(1) == yy).float().mean().item(),
                          F.cross_entropy(lg, yy).item())
        out["ALL"] = tuple(np.mean([out[d][j] for d in self.ds_list]) for j in range(2))
        return out

# ================================================================ workers
class TrainWorker(QThread):
    progress = Signal(int, int, str); done = Signal()
    def __init__(self, core): super().__init__(); self.core = core
    def run(self):
        self.core.train_all(lambda i, n, nm: self.progress.emit(i, n, nm)); self.done.emit()

class OverlayWorker(QThread):
    row_ready = Signal(int, int, np.ndarray, float, float); done = Signal(int)
    def __init__(self, job, core, mean, E0, E1, extent, res, ds):
        super().__init__()
        self.job, self.core = job, core
        self.mean, self.E0, self.E1 = mean, E0, E1
        self.extent, self.res, self.ds = extent, res, ds
        self.stop = False
    def run(self):
        lo1, hi1, lo2, hi2 = self.extent
        xs, ys = np.linspace(lo1, hi1, self.res), np.linspace(lo2, hi2, self.res)
        vmin = vmax = None
        for r, t in enumerate(ys):
            if self.stop: return
            row = np.array([self.core.metrics(self.mean + s*self.E0 + t*self.E1)[self.ds][0]
                            for s in xs])
            vmin = float(row.min()) if vmin is None else min(vmin, float(row.min()))
            vmax = float(row.max()) if vmax is None else max(vmax, float(row.max()))
            self.row_ready.emit(self.job, r, row, vmin, vmax)
        self.done.emit(self.job)

def ensure_nets(core):
    if os.path.exists(CACHE):
        core.load_from_cache(); return
    dlg = QProgressDialog("Training destination networks…", None, 0, 1)
    dlg.setWindowModality(Qt.ApplicationModal); dlg.setMinimumWidth(380); dlg.show()
    w = TrainWorker(core)
    w.progress.connect(lambda i, n, nm: (dlg.setMaximum(n), dlg.setValue(i),
                                         dlg.setLabelText(f"training {nm} …")))
    w.start()
    if not w.isFinished():
        loop = QEventLoop(); w.done.connect(loop.quit); loop.exec()
    w.wait(); dlg.close(); dlg.deleteLater()

# ================================================================ widgets
class LabeledSlider(QWidget):
    valueChangedF = Signal(float)
    def __init__(self, text, lo, hi, val, dec=2, parent=None):
        super().__init__(parent)
        self.dec, self.scale, self.lo, self.hi = dec, 10**dec, lo, hi
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.lab = QLabel(text); self.lab.setMinimumWidth(108)
        self.sl = QSlider(Qt.Horizontal); self.sl.setFocusPolicy(Qt.NoFocus)
        self.sl.setRange(int(lo*self.scale), int(hi*self.scale))
        self.sl.setValue(int(val*self.scale))
        self.val_lab = QLabel(f"{val:.{dec}f}"); self.val_lab.setMinimumWidth(62)
        self.val_lab.setFont(QFont("monospace", 8))
        lay.addWidget(self.lab); lay.addWidget(self.sl, 1); lay.addWidget(self.val_lab)
        self.sl.valueChanged.connect(self._emit)
    def _emit(self, v):
        f = v / self.scale
        self.val_lab.setText(f"{f:.{self.dec}f}"); self.valueChangedF.emit(f)
    def value(self): return self.sl.value() / self.scale
    def set_value(self, v, silent=True):
        if silent: self.sl.blockSignals(True)
        self.sl.setValue(int(round(v * self.scale)))
        if silent:
            self.sl.blockSignals(False)
            self.val_lab.setText(f"{v:.{self.dec}f}")
    def set_range(self, lo, hi):
        self.lo, self.hi = lo, hi
        self.sl.setRange(int(lo*self.scale), int(hi*self.scale))

class HBars(QWidget):
    """Horizontal bars, signed-capable, dynamic labels (composition/attribution)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.labels, self.vals, self.caption = [], np.zeros(0), ""
        self.setMinimumHeight(70)
    def set_data(self, labels, vals, caption=""):
        self.labels, self.vals, self.caption = list(labels), np.asarray(vals), caption
        self.setMinimumHeight(max(46, 15 * len(labels) + 18)); self.update()
    def paintEvent(self, ev):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.setPen(QColor("#888"))
        p.drawText(QRectF(4, 1, w-8, 13), Qt.AlignLeft, self.caption)
        if not len(self.vals): return
        n = len(self.vals); bh = (h - 20) / n
        x0, x1 = 78.0, w - 46.0
        mx = max(np.abs(self.vals).max(), 1e-9)
        mid = x0 + (x1 - x0) * 0.5 if self.vals.min() < 0 else x0
        span = (x1 - mid) if self.vals.min() < 0 else (x1 - x0)
        p.setPen(QColor("#555"))
        if self.vals.min() < 0: p.drawLine(QPointF(mid, 16), QPointF(mid, h - 4))
        for i, v in enumerate(self.vals):
            y = 16 + i * bh
            p.setPen(QColor("#777"))
            p.drawText(QRectF(0, y, x0 - 4, bh), Qt.AlignVCenter | Qt.AlignRight,
                       self.labels[i])
            L = span * float(v) / mx
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#22d3ee") if v >= 0 else QColor("#f87171"))
            p.drawRect(QRectF(mid if v < 0 else mid, y + bh*0.18, L, bh*0.64)
                       if v >= 0 else QRectF(mid + L, y + bh*0.18, -L, bh*0.64))
            p.setPen(QColor("#999"))
            p.drawText(QRectF(w - 44, y, 42, bh), Qt.AlignVCenter, f"{v:+.2f}")

# ================================================================ main window
class MainWindow(QMainWindow):
    def __init__(self, core: Core):
        super().__init__()
        self.core = core
        self.names = [n for n in core.nets if not n.startswith("const:")]
        self.M = np.stack([core.nets[n] for n in self.names])
        self.N_PARAMS = self.M.shape[1]
        self.SHAPES = [tuple(p.shape) for p in core.model.parameters()]
        self.tsl, i = [], 0
        for sh in self.SHAPES:
            n = int(np.prod(sh)); self.tsl.append(slice(i, i + n)); i += n
        self.tlabels = ["W1", "b1", "W2", "b2", "W3", "b3"]

        self.fleet_gen, self.fleet_order, self.fleet_vecs = {}, [], {}
        for nm, g in [("const:0.1", sat_const(0.1)), ("const:0.5", sat_const(0.5)),
                      ("const:0.9", sat_const(0.9))] + \
                     [(f"prng:s{s}", sat_prng(s*111)) for s in (1, 2, 3, 4)] + \
                     [("ramp", sat_ramp(-.5, .5)), ("sin:k3", sat_sin(3, .3)),
                      ("sign:k7", sat_sign(7, .2)), ("onehot:L1", sat_onehot(0, .25))]:
            self._add_sat(nm, g)
        self.active = list(self.fleet_order)
        self.prng_seed, self.pattern_i = 5, 0
        self.pattern_cycle = [("sin", sat_sin(5, .3)), ("sign", sat_sign(13, .2)),
                              ("ramp", sat_ramp(.6, -.6)), ("onehot", sat_onehot(2, .2))]

        self.frame = None
        self.dest, self.route, self.step_i = None, None, 0
        self.driving, self.dest_err = False, 0.0
        self.theta_ref = None
        self.snaps, self.snap_artists = [], []
        self.overlay_worker, self.overlay_job = None, 0
        self.ready = False

        self._build_ui()
        self._wire()
        self._draw_static_map()
        self.ready = True
        self._rebuild_frame(init_name="gen:ALL")
        self._log(f"ready — {len(self.names)} nets, {len(self.active)} anchors, "
                  f"N={self.N_PARAMS:,}, elevator armed")
        self._do_live()

    # ---------------- fleet
    def _add_sat(self, name, gen):
        if name in self.fleet_gen or len(self.fleet_order) >= MAX_FLEET: return False
        self.fleet_gen[name] = gen
        self.fleet_order.append(name)
        self.fleet_vecs[name] = gen(self.N_PARAMS, self.SHAPES)
        return True

    def _add_prng(self):
        nm = f"prng:s{self.prng_seed}"
        if self._add_sat(nm, sat_prng(self.prng_seed*111 + 7)):
            self.prng_seed += 1
            self._refresh_con_ui(); self._toggle_active(nm, True)

    def _add_pattern(self):
        kind, gen = self.pattern_cycle[self.pattern_i % len(self.pattern_cycle)]
        self.pattern_i += 1
        nm = f"{kind}:x{self.pattern_i}"
        if self._add_sat(nm, gen):
            self._refresh_con_ui(); self._toggle_active(nm, True)

    # ---------------- frame + elevator
    def _rebuild_frame(self, keep_vec=None, init_name=None):
        acts = [n for n in self.fleet_order if n in self.active]
        K = len(acts)
        if K < 4:
            self.statusBar().showMessage("⚠ ≥4 satellites required for a fix", 6000)
            return False
        X = np.stack([self.fleet_vecs[n] for n in acts])
        Xc = X - X.mean(0); k = K - 1
        if np.linalg.matrix_rank(Xc) < k:
            self.statusBar().showMessage(
                "⚠ rank-deficient constellation — constants are collinear (c·1).", 9000)
            return False
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        E, mean = Vt[:k], X.mean(0)
        var = S[:k]**2 / (S**2).sum()
        cond = float(S[0] / max(S[k-1], 1e-300))
        anc_c, net_c = (X-mean) @ E.T, (self.M-mean) @ E.T
        perp = self.M - mean - ((self.M-mean) @ E.T) @ E
        alt_max = float(np.linalg.norm(perp, axis=1).max())
        allc = np.vstack([anc_c, net_c]); pad = 0.08
        lo = allc.min(0) - pad*np.ptp(allc, 0) - 1e-3
        hi = allc.max(0) + pad*np.ptp(allc, 0) + 1e-3
        D = X - mean
        self.frame = dict(sats=acts, B=X, mean=mean, E=E, var=var, cond=cond,
                          anc_c=anc_c, net_c=net_c, lo=lo, hi=hi, D=D,
                          Dpinv=np.linalg.pinv(D), alt_max=alt_max, n=None)
        self.extent = (float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]))

        if self.driving: self._toggle_drive()
        self.dest, self.route = None, None
        self.route_line.set_data([], []); self.dest_mark.set_data([], [])
        self.steps.clear()
        cur = keep_vec if keep_vec is not None else \
            (self.core.nets.get(init_name) if init_name else mean)
        self._set_nhat(cur)
        self._rebuild_axis_sliders(cur)
        self._restyle_map(); self._update_chart(); self._update_section()
        self._log(f"fix: {k}-D chart + elevator from {K} anchors · κ={cond:.0f} · "
                  f"var(c1,c2)={var[0]+var[1]:.0%} · max net altitude {alt_max:.2f}")
        return True

    def _set_nhat(self, preserve_vec=None):
        f = self.frame
        txt = self.el_combo.currentText()
        if txt.startswith("→ dest"):
            v = self.core.nets[self.dest_combo.currentText()]
        elif txt.startswith("→ nearest"):
            ref = preserve_vec if preserve_vec is not None else f["mean"]
            v = self.M[int(np.linalg.norm(self.M - ref, axis=1).argmin())]
        else:
            v = np.random.RandomState(123).standard_normal(self.N_PARAMS)
        n = v - f["mean"] - ((v - f["mean"]) @ f["E"].T) @ f["E"]
        nn = np.linalg.norm(n)
        f["n"] = n / nn if nn > 1e-9 else None
        self.el_range = 1.35 * f["alt_max"] + 0.5

    def _on_elevator_dir(self):
        if not self.ready or self.frame is None: return
        vec = self._cur_vec()
        self._set_nhat(vec)
        h = float((vec - self.frame["mean"]) @ self.frame["n"]) \
            if self.frame["n"] is not None else 0.0
        self.el_slider.set_range(-self.el_range, self.el_range)
        self.el_slider.set_value(np.clip(h, -self.el_range, self.el_range))
        self._update_section(); self.schedule_live()

    def _rebuild_axis_sliders(self, cur_vec):
        lay = self.axis_lay
        while lay.count():
            it = lay.takeAt(0); w = it.widget()
            if w: w.deleteLater()
        self.pos_sliders = []
        f = self.frame
        c0 = (cur_vec - f["mean"]) @ f["E"].T
        h0 = float((cur_vec - f["mean"]) @ f["n"]) if f["n"] is not None else 0.0
        for i in range(len(f["E"])):
            s = LabeledSlider(f"c{i+1} ({f['var'][i]:4.0%})",
                              float(f["lo"][i]), float(f["hi"][i]), float(c0[i]))
            s.valueChangedF.connect(self._on_user_move)
            lay.addWidget(s); self.pos_sliders.append(s)
        self.el_slider = LabeledSlider("h (elevator)", -self.el_range, self.el_range,
                                       float(np.clip(h0, -self.el_range, self.el_range)))
        self.el_slider.valueChangedF.connect(self._on_user_move)
        lay.addWidget(self.el_slider); self.pos_sliders.append(self.el_slider)
        cur_ax = self.ax_combo.currentIndex()
        self.ax_combo.blockSignals(True); self.ax_combo.clear()
        self.ax_combo.addItems([f"c{i+1}" for i in range(len(f["E"]))] + ["h (elev)"])
        self.ax_combo.setCurrentIndex(min(max(cur_ax, 0), len(f["E"])))
        self.ax_combo.blockSignals(False)

    def _restyle_map(self):
        f = self.frame
        self.ax.set_xlim(self.extent[0], self.extent[1])
        self.ax.set_ylim(self.extent[2], self.extent[3])
        self.im.set_extent(list(self.extent))
        ac = f["anc_c"]
        self.sat_scatter.set_offsets(ac[:, :2])
        off = [n for n in self.fleet_order if n not in f["sats"]]
        if off:
            oc = np.array([(self.fleet_vecs[n]-f["mean"]) @ f["E"].T[:2] for n in off])
            self.off_scatter.set_offsets(oc); self.off_scatter.set_visible(True)
        else:
            self.off_scatter.set_visible(False)
        self.net_scatter.set_offsets(f["net_c"][:, :2])
        for a in self.anns:
            try: a.remove()
            except Exception: pass
        self.anns = [self.ax.annotate(n, (x, y), fontsize=5.5, color="k",
                                      xytext=(3, 3), textcoords="offset points")
                     for n, (x, y) in zip(self.names, f["net_c"][:, :2])]
        self.anns += [self.ax.annotate("▲"+n, (x, y), fontsize=6, color="#b91c1c",
                                       xytext=(4, -9), textcoords="offset points")
                      for n, (x, y) in zip(f["sats"], ac[:, :2])]
        self.canvas.draw_idle()

    # ---------------- fix (multilateration + elevator decomposition)
    def _fix(self, vec):
        f = self.frame; B = f["B"]
        d = np.linalg.norm(B - vec, axis=1)
        rows = 2.0 * ((B[0] - B[1:]) @ f["E"].T)
        rhs = d[1:]**2 - d[0]**2 - (B[1:]**2).sum(1) + (B[0] @ B[0])
        try:    c = np.linalg.solve(rows, rhs); res = float(np.abs(rows @ c - rhs).max())
        except np.linalg.LinAlgError:
                c = np.linalg.lstsq(rows, rhs, rcond=None)[0]
                res = float(np.abs(rows @ c - rhs).max())
        proj = f["mean"] + c @ f["E"]
        eps = float(np.linalg.norm(vec - proj))
        h = float((vec - proj) @ f["n"]) if f["n"] is not None else 0.0
        return c, eps, d, h, res

    def vec_of(self, c, h=0.0):
        f = self.frame
        return f["mean"] + np.asarray(c) @ f["E"] + h * (f["n"] or 0.0)

    def _cur_ch(self):
        c = np.array([s.value() for s in self.pos_sliders[:-1]])
        return c, self.el_slider.value()
    def _cur_vec(self):
        c, h = self._cur_ch(); return self.vec_of(c, h)

    def _composition(self, c):
        f = self.frame
        lam = np.asarray(c) @ f["E"] @ f["Dpinv"]
        return lam + (1.0 - lam.sum()) / len(f["sats"])

    # ---------------- UI
    def _build_ui(self):
        self.setWindowTitle(f"GPS-θ v8 · weight fields + elevator · {self.N_PARAMS:,} params")
        self.resize(1680, 1000)
        mb = self.menuBar(); m = mb.addMenu("&File")
        for text, slot in (("Save chart &PNG…", self.save_png),
                           ("&Export journal…", self.export_journal),
                           ("Copy &methods paragraph", self.copy_methods)):
            a = QAction(text, self); a.triggered.connect(slot); m.addAction(a)
        aq = QAction("&Quit", self); aq.triggered.connect(self.close); m.addAction(aq)

        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central); root.setContentsMargins(4, 4, 4, 4)
        hsplit = QSplitter(Qt.Horizontal); root.addWidget(hsplit, 1)

        # left: chart + section (+ section controls)
        left = QSplitter(Qt.Vertical)
        self.fig = Figure(figsize=(5.6, 4.0), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111)
        left.addWidget(self.canvas)
        secw = QWidget(); sv = QVBoxLayout(secw); sv.setContentsMargins(2, 2, 2, 2)
        self.sfig = Figure(figsize=(5.6, 2.3), dpi=100)
        self.scanvas = FigureCanvasQTAgg(self.sfig)
        self.sax = self.sfig.add_subplot(111)
        sv.addWidget(self.scanvas)
        srow = QHBoxLayout()
        srow.addWidget(QLabel("sweep"))
        self.ax_combo = QComboBox(); srow.addWidget(self.ax_combo)
        self.strace_combo = QComboBox()
        self.strace_combo.addItems(["component", "tensor norms", "pseudoranges", "||theta||"])
        srow.addWidget(self.strace_combo)
        srow.addWidget(QLabel("θ#"))
        self.scomp = QSpinBox(); self.scomp.setRange(0, self.N_PARAMS - 1); srow.addWidget(self.scomp)
        self.btn_sweep = QPushButton("sweep ▶"); self.btn_sweep.setFocusPolicy(Qt.NoFocus)
        srow.addWidget(self.btn_sweep)
        sv.addLayout(srow)
        left.addWidget(secw); left.setSizes([430, 300])
        hsplit.addWidget(left)

        # middle: morphology + readouts
        mid = QWidget(); mv = QVBoxLayout(mid); mv.setContentsMargins(2, 2, 2, 2)
        mv.addWidget(QLabel("§3.4 morphology — click a pixel to plot that weight below"))
        self.wfig = Figure(figsize=(4.6, 1.9), dpi=100)
        self.wcanvas = FigureCanvasQTAgg(self.wfig)
        self.wfig.subplots_adjust(left=.04, right=.99, top=.86, bottom=.14, wspace=.25)
        gs = self.wfig.add_gridspec(1, 4, width_ratios=[2, 1, 0.7, 0.32])
        self.waxes, self.wims = [], []
        for gi, (nm, sh) in enumerate(zip(["W1", "W2", "W3"],
                                          [self.SHAPES[0], self.SHAPES[2], self.SHAPES[4]])):
            axw = self.wfig.add_subplot(gs[0, gi])
            im = axw.imshow(np.zeros(sh), cmap="RdBu_r", aspect="auto",
                            vmin=-.2, vmax=.2, interpolation="nearest")
            axw.set_title(nm, fontsize=7); axw.set_xticks([]); axw.set_yticks([])
            self.waxes.append(axw); self.wims.append(im)
        axb = self.wfig.add_subplot(gs[0, 3])
        self.wims.append(axb.imshow(np.zeros((1, 74)), cmap="RdBu_r", aspect="auto",
                                    vmin=-.2, vmax=.2, interpolation="nearest"))
        axb.set_title("biases", fontsize=7); axb.set_xticks([]); axb.set_yticks([])
        wrow = QHBoxLayout()
        wrow.addWidget(QLabel("view")); self.wview = QComboBox()
        self.wview.addItems(["θ", "θ − mean", "θ − nearest net", "θ − dest"])
        wrow.addWidget(self.wview, 1)
        mv.addWidget(self.wcanvas); mv.addLayout(wrow)

        self.info = QTextEdit(); self.info.setReadOnly(True)
        self.info.setFont(QFont("monospace", 8)); self.info.setMaximumHeight(84)
        self.signal = QTextEdit(); self.signal.setReadOnly(True)
        self.signal.setFont(QFont("monospace", 7.5)); self.signal.setMaximumHeight(108)
        self.compo = HBars()
        self.attrib = HBars()
        self.steps = QTextEdit(); self.steps.setReadOnly(True)
        self.steps.setFont(QFont("monospace", 8)); self.steps.setMaximumHeight(96)
        for lab in ("§4.1 FIX (ranges → c, h, ε, residual)", "§4.1 pseudoranges ρᵢ",
                    "composition θ̂ = Σλᵢ·satᵢ (in-span, Σλ=1)",
                    "per-tensor transit attribution", "§4.3 transit"):
            l = QLabel(lab); l.setStyleSheet("font-size:7.5pt; color:#555;")
            mv.addWidget(l)
        mv.addWidget(self.info); mv.addWidget(self.signal)
        mv.addWidget(self.compo); mv.addWidget(self.attrib); mv.addWidget(self.steps)
        hsplit.addWidget(mid)

        # right: journal
        jr = QWidget(); jv = QVBoxLayout(jr); jv.setContentsMargins(2, 2, 2, 2)
        jv.addWidget(QLabel("§5 journal"))
        self.journal = QTextEdit(); self.journal.setReadOnly(True)
        self.journal.setFont(QFont("monospace", 7.5))
        jv.addWidget(self.journal)
        hsplit.addWidget(jr)
        hsplit.setSizes([560, 470, 270])

        # bottom controls
        bottom = QWidget(); bh = QHBoxLayout(bottom); bh.setContentsMargins(2, 2, 2, 2)

        g_con = QGroupBox("§3.1 Constellation"); cv = QVBoxLayout(g_con)
        hold = QWidget(); self.con_lay = QVBoxLayout(hold)
        self.con_lay.setContentsMargins(2, 2, 2, 2); self.con_lay.setSpacing(1)
        cs = QScrollArea(); cs.setWidget(hold); cs.setWidgetResizable(True)
        cs.setMinimumHeight(120); cs.setMaximumHeight(200)
        cv.addWidget(cs)
        r = QHBoxLayout()
        b1 = QPushButton("＋PRNG"); b2 = QPushButton("＋pattern")
        for b in (b1, b2): b.setFocusPolicy(Qt.NoFocus)
        b1.clicked.connect(self._add_prng); b2.clicked.connect(self._add_pattern)
        r.addWidget(b1); r.addWidget(b2); cv.addLayout(r)
        bh.addWidget(g_con)

        g_pos = QGroupBox("§3.3 Position (chart + elevator)"); pv = QVBoxLayout(g_pos)
        hold2 = QWidget(); self.axis_lay = QVBoxLayout(hold2)
        self.axis_lay.setContentsMargins(2, 2, 2, 2); self.axis_lay.setSpacing(3)
        ps = QScrollArea(); ps.setWidget(hold2); ps.setWidgetResizable(True)
        ps.setMinimumHeight(120); ps.setMaximumHeight(200)
        pv.addWidget(ps)
        r_e = QHBoxLayout(); r_e.addWidget(QLabel("elevator →"))
        self.el_combo = QComboBox()
        self.el_combo.addItems(["→ dest (exact landing)", "→ nearest net", "fixed random"])
        r_e.addWidget(self.el_combo, 1); pv.addLayout(r_e)
        pv.addWidget(QLabel("click chart = set c1,c2 · near ▲/□ = jump"))
        bh.addWidget(g_pos, 2)

        g_tr = QGroupBox("§4.3 Transit"); tv = QVBoxLayout(g_tr)
        r1 = QHBoxLayout(); r1.addWidget(QLabel("dest"))
        self.dest_combo = QComboBox(); self.dest_combo.addItems(self.names)
        r1.addWidget(self.dest_combo, 1); tv.addLayout(r1)
        r2 = QHBoxLayout(); r2.addWidget(QLabel("shape"))
        self.shape_combo = QComboBox()
        self.shape_combo.addItems(["L-path", "direct", "via anchor"])
        r2.addWidget(self.shape_combo, 1); tv.addLayout(r2)
        r3 = QHBoxLayout()
        self.btn_nav = QPushButton("Navigate"); self.btn_drive = QPushButton("Drive")
        self.btn_cancel = QPushButton("Cancel"); self.btn_snap = QPushButton("snap")
        for b in (self.btn_nav, self.btn_drive, self.btn_cancel, self.btn_snap):
            b.setFocusPolicy(Qt.NoFocus); r3.addWidget(b)
        tv.addLayout(r3)
        r4 = QHBoxLayout()
        self.btn_wadd = QPushButton("＋snapshot"); self.btn_wdel = QPushButton("−")
        self.btn_ref = QPushButton("ref=here")
        for b in (self.btn_wadd, self.btn_wdel, self.btn_ref):
            b.setFocusPolicy(Qt.NoFocus); r4.addWidget(b)
        tv.addLayout(r4)
        self.speed = LabeledSlider("speed (u/s)", 1, 60, 15, dec=0)
        tv.addWidget(self.speed)
        bh.addWidget(g_tr)

        g_ch = QGroupBox("§3.2 Chart — weight fields"); chv = QVBoxLayout(g_ch)
        self.field_combo = QComboBox(); self.field_combo.addItems(FIELDS)
        chv.addWidget(self.field_combo)
        r5 = QHBoxLayout(); r5.addWidget(QLabel("ref"))
        self.ref_combo = QComboBox(); r5.addWidget(self.ref_combo, 1)
        chv.addLayout(r5)
        r6 = QHBoxLayout(); r6.addWidget(QLabel("res"))
        self.res_spin = QSpinBox(); self.res_spin.setRange(21, 121)
        self.res_spin.setSingleStep(10); self.res_spin.setValue(61)
        r6.addWidget(self.res_spin)
        self.contour_chk = QCheckBox("contours"); self.contour_chk.setChecked(True)
        r6.addWidget(self.contour_chk); chv.addLayout(r6)
        self.ov_check = QCheckBox("functional overlay (accuracy) — demoted")
        self.ov_check.setStyleSheet("font-size:7.5pt;")
        chv.addWidget(self.ov_check)
        r7 = QHBoxLayout(); r7.addWidget(QLabel("overlay ds"))
        self.ov_ds = QComboBox(); self.ov_ds.addItems(["ALL"] + self.core.ds_list)
        r7.addWidget(self.ov_ds, 1); chv.addLayout(r7)
        self.btn_gallery = QPushButton("field gallery ⧉")
        self.btn_gallery.setFocusPolicy(Qt.NoFocus)
        chv.addWidget(self.btn_gallery)
        bh.addWidget(g_ch)

        g_st = QGroupBox("§4.2 Studies (weight-only)"); stv = QVBoxLayout(g_st)
        for text, slot in (("coverage (R², altitude vs K)", self.study_coverage),
                           ("GDOP report", self.study_gdop),
                           ("constellation geometry", self.study_geometry),
                           ("copy methods ¶", self.copy_methods)):
            b = QPushButton(text); b.setFocusPolicy(Qt.NoFocus)
            b.clicked.connect(slot); stv.addWidget(b)
        stv.addStretch(1)
        bh.addWidget(g_st)
        root.addWidget(bottom)

        self.live_timer = QTimer(self); self.live_timer.setSingleShot(True)
        self.live_timer.setInterval(30); self.live_timer.timeout.connect(self._do_live)
        self.drive_timer = QTimer(self); self.drive_timer.setInterval(60)
        self.drive_timer.timeout.connect(self._drive_tick)
        self.recalc_timer = QTimer(self); self.recalc_timer.setSingleShot(True)
        self.recalc_timer.setInterval(350)
        self.recalc_timer.timeout.connect(lambda: self.dest and self.navigate())
        self.sweep_timer = QTimer(self); self.sweep_timer.setInterval(70)
        self.sweep_timer.timeout.connect(self._sweep_tick)
        self.sweep_dir = 1
        hint = QLabel(" Space drive · N nav · R snap · S snapshot · X un-snap · ←→↑↓ nudge ")
        self.statusBar().addPermanentWidget(hint)
        self._refresh_con_ui()

    def _refresh_con_ui(self):
        while self.con_lay.count():
            it = self.con_lay.takeAt(0); w = it.widget()
            if w: w.deleteLater()
        for name in self.fleet_order:
            cb = QCheckBox(("▲ " if name in CORE_SATS else "   ") + name)
            cb.setChecked(name in self.active)
            cb.setStyleSheet("font-size:7.5pt;")
            cb.toggled.connect(lambda on, n=name: self._toggle_active(n, on))
            self.con_lay.addWidget(cb)
        cur = self.ref_combo.currentText()
        self.ref_combo.blockSignals(True); self.ref_combo.clear()
        self.ref_combo.addItems(self.fleet_order)
        if cur in self.fleet_order: self.ref_combo.setCurrentText(cur)
        self.ref_combo.blockSignals(False)

    def _wire(self):
        self.btn_nav.clicked.connect(self.navigate)
        self.btn_drive.clicked.connect(self._toggle_drive)
        self.btn_cancel.clicked.connect(self.cancel_route)
        self.btn_snap.clicked.connect(self.snap_nearest)
        self.btn_wadd.clicked.connect(self.add_snapshot)
        self.btn_wdel.clicked.connect(self.del_snapshot)
        self.btn_ref.clicked.connect(self._set_ref_here)
        self.btn_gallery.clicked.connect(self.open_gallery)
        self.btn_sweep.clicked.connect(self._toggle_sweep)
        self.el_combo.currentTextChanged.connect(lambda _: self._on_elevator_dir())
        self.dest_combo.currentTextChanged.connect(self._on_dest_changed)
        self.field_combo.currentIndexChanged.connect(lambda _: self._update_chart())
        self.ref_combo.currentIndexChanged.connect(lambda _: self._update_chart())
        self.res_spin.valueChanged.connect(lambda _: self._update_chart())
        self.contour_chk.toggled.connect(lambda _: self._update_chart())
        self.ov_check.toggled.connect(self._on_overlay_toggle)
        self.ov_ds.currentTextChanged.connect(lambda _: self._on_overlay_toggle(True))
        self.wview.currentIndexChanged.connect(lambda _: self._update_viewer())
        self.ax_combo.currentIndexChanged.connect(lambda _: self._update_section())
        self.strace_combo.currentIndexChanged.connect(lambda _: self._update_section())
        self.scomp.valueChanged.connect(lambda _: self._update_section())
        self.canvas.mpl_connect("button_press_event", self.on_chart_click)
        self.wcanvas.mpl_connect("button_press_event", self.on_morph_click)
        for key, slot in (("Space", self._toggle_drive), ("N", self.navigate),
                          ("R", self.snap_nearest), ("S", self.add_snapshot),
                          ("X", self.del_snapshot),
                          ("Left", lambda: self._nudge(0, -1)),
                          ("Right", lambda: self._nudge(0, +1)),
                          ("Up", lambda: self._nudge(1, +1)),
                          ("Down", lambda: self._nudge(1, -1))):
            sc = QShortcut(QKeySequence(key), self); sc.activated.connect(slot)

    def _nudge(self, axis, sgn):
        if not self.ready or axis >= len(self.pos_sliders): return
        sl = self.pos_sliders[axis]
        sl.set_value(sl.value() + sgn * 0.01 * (sl.hi - sl.lo), silent=False)

    def _toggle_active(self, name, on):
        if on: self.active.append(name)
        elif name in self.active: self.active.remove(name)
        keep = self._cur_vec() if (self.frame and self.pos_sliders) else None
        if not self._rebuild_frame(keep_vec=keep):
            if on and name in self.active: self.active.remove(name)
            elif not on: self.active.append(name)
            self._refresh_con_ui()

    def _on_dest_changed(self, _=None):
        if self.el_combo.currentText().startswith("→ dest"):
            vec = self._cur_vec()
            self._set_nhat(vec)
            if self.frame["n"] is not None:
                h = float((vec - self.frame["mean"]) @ self.frame["n"])
                self.el_slider.set_range(-self.el_range, self.el_range)
                self.el_slider.set_value(np.clip(h, -self.el_range, self.el_range))
        self._update_viewer(); self.schedule_live()

    def _log(self, msg): self.journal.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def _set_ref_here(self):
        self.theta_ref = self._cur_vec()
        self._log("attribution reference set to current position")
        self.schedule_live()

    # ---------------- map artists
    def _draw_static_map(self):
        cmap = mpl.colormaps["viridis"].copy(); cmap.set_bad("0.12")
        self.im = self.ax.imshow(np.full((21, 21), np.nan), origin="lower",
                                 extent=[-1, 1, -1, 1], aspect="auto",
                                 cmap=cmap, vmin=0., vmax=1., zorder=0)
        self.cbar = self.fig.colorbar(self.im, ax=self.ax, pad=0.02)
        self.cbar.set_label("weight field")
        self.sat_scatter = self.ax.scatter([], [], marker="^", s=110, c="red", zorder=6)
        self.off_scatter = self.ax.scatter([], [], marker="x", s=26, c="#666", zorder=5)
        self.net_scatter = self.ax.scatter([], [], s=20, c="#dddddd", zorder=5)
        self.anns = []
        self.dest_mark, = self.ax.plot([], [], "*", ms=16, color="#f59e0b", zorder=7)
        self.route_line, = self.ax.plot([], [], "-", color="#22d3ee", lw=2, alpha=.9, zorder=6)
        self.dot, = self.ax.plot([], [], "w+", ms=14, mew=2, zorder=8)
        self.htext = self.ax.annotate("", (0.02, 0.02), xycoords="axes fraction",
                                      fontsize=7, color="#f59e0b", zorder=9)

    # ---------------- §3.2 field computation (shared by chart + gallery)
    def _compute_field(self, kind, S, T):
        f, cf = self.frame, getattr(self, "cf", None)
        m, E0, E1 = f["mean"], f["E"][0], f["E"][1]
        n0 = float(m @ m); n1 = float(m @ E0); n2 = float(m @ E1)
        norm2 = n0 + 2*n1*S + 2*n2*T + S*S + T*T
        B = f["B"]
        anc_lin = np.stack([[float(a @ m), float(a @ E0), float(a @ E1)] for a in B])
        anc_q = (B**2).sum(1)
        net_lin = np.stack([[float(v @ m), float(v @ E0), float(v @ E1)] for v in self.M])
        net_q = (self.M**2).sum(1)
        if kind == FIELDS[0] or kind == FIELDS[3]:
            i = self.ref_combo.currentIndex()
            lin, q = anc_lin[i], anc_q[i]
            dot = lin[0] + lin[1]*S + lin[2]*T
            if kind == FIELDS[0]:
                return (np.sqrt(np.maximum(norm2 - 2*dot + q, 0)), "viridis", None,
                        f"ρ(θ, {f['sats'][i]}) — exact circles")
            return (dot / (np.sqrt(norm2) * np.sqrt(q) + 1e-300), "RdBu_r", (-1, 1),
                    f"cos(θ, {f['sats'][i]})")
        if kind == FIELDS[1]:
            i = max(self.ref_combo.currentIndex(), 1)
            l1, li = anc_lin[0], anc_lin[i]
            G = 2*((l1[0]-li[0]) + (l1[1]-li[1])*S + (l1[2]-li[2])*T) + anc_q[i] - anc_q[0]
            return (G, "viridis", None,
                    f"GPS linear eq Δρ²({f['sats'][i]} vs SAT1) — straight contours")
        if kind == FIELDS[2]:
            return (np.sqrt(np.maximum(norm2, 0)), "viridis", None, "‖θ(s,t)‖")
        if kind in (FIELDS[4], FIELDS[5]):
            d2 = np.stack([norm2 - 2*(l[0] + l[1]*S + l[2]*T) + q
                           for l, q in zip(net_lin, net_q)])
            if kind == FIELDS[4]:
                return (np.sqrt(np.maximum(d2.min(0), 0)), "viridis", None,
                        "minₘ ‖θ−θₘ‖ (Voronoi distance)")
            return (d2.argmin(0).astype(float), "tab20",
                    (-.5, len(self.names) - .5), "Voronoi cells of trained nets")
        j = (self.ref_combo.currentIndex() % 3) * 2
        J = self.tsl[j]
        mj, e0j, e1j = m[J], E0[J], E1[J]
        G = np.sqrt(np.maximum(mj@mj + 2*(mj@e0j)*S + 2*(mj@e1j)*T
                               + (e0j@e0j)*S*S + 2*(e0j@e1j)*S*T + (e1j@e1j)*T*T, 0))
        return (G, "viridis", None, f"‖θ_( {self.tlabels[j]} )‖")

    def _update_chart(self):
        if not self.ready or self.frame is None: return
        kind = self.field_combo.currentText()
        res = self.res_spin.value()
        lo1, hi1, lo2, hi2 = self.extent
        S, T = np.meshgrid(np.linspace(lo1, hi1, res), np.linspace(lo2, hi2, res))
        G, cmap, clim, title = self._compute_field(kind, S, T)
        self.ax.set_title(title, fontsize=9)
        if getattr(self, "contour", None) is not None:
            self.contour.remove(); self.contour = None
        if getattr(self, "im", None) is not None:
            self.im.remove()
            try: self.cbar.remove()
            except Exception: pass
        self.im = self.ax.imshow(G, origin="lower", extent=list(self.extent),
                                 aspect="auto", cmap=cmap, zorder=0,
                                 interpolation="nearest")
        if clim: self.im.set_clim(*clim)
        self.cbar = self.fig.colorbar(self.im, ax=self.ax, pad=0.02)
        self.cbar.set_label(kind)
        if self.contour_chk.isChecked() and kind != FIELDS[5]:
            self.contour = self.ax.contour(S, T, G, levels=8, colors="w",
                                           linewidths=.4, alpha=.45, zorder=1)
        self._ov_extent_sync()
        for a in self.anns: a.set_zorder(7)
        self.sat_scatter.set_zorder(8); self.net_scatter.set_zorder(8)
        self.dot.set_zorder(10); self.route_line.set_zorder(9)
        self.canvas.draw_idle()
        if self.ov_check.isChecked(): self._start_overlay()

    def _ov_extent_sync(self):
        if getattr(self, "_overlay_im", None) is not None:
            self._overlay_im.set_extent(list(self.extent))

    # ---------------- overlay (optional functional field)
    def _on_overlay_toggle(self, on):
        if not self.ready: return
        if not self.ov_check.isChecked():
            if getattr(self, "_overlay_im", None) is not None:
                self._overlay_im.remove(); self._overlay_im = None
            self.canvas.draw_idle(); return
        self._start_overlay()

    def _start_overlay(self):
        if self.overlay_worker and self.overlay_worker.isRunning():
            self.overlay_worker.stop = True
        self.overlay_job += 1
        f = self.frame
        w = OverlayWorker(self.overlay_job, self.core, f["mean"], f["E"][0], f["E"][1],
                          self.extent, self.res_spin.value(), self.ov_ds.currentText())
        w.row_ready.connect(self._on_ov_row); w.done.connect(self._on_ov_done)
        self.overlay_worker = w; w.start()

    def _on_ov_row(self, job, r, row, vmin, vmax):
        if job != self.overlay_job: return
        res = self.res_spin.value()
        if getattr(self, "_ovH", None) is None or self._ovH.shape != (res, res):
            self._ovH = np.full((res, res), np.nan)
        self._ovH[r] = row
        if getattr(self, "_overlay_im", None) is None:
            self._overlay_im = self.ax.imshow(self._ovH, origin="lower",
                                              extent=list(self.extent), aspect="auto",
                                              cmap="inferno", alpha=.42, zorder=2)
        else:
            self._overlay_im.set_data(self._ovH)
        self._overlay_im.set_clim(vmin, vmax)
        self.canvas.draw_idle()

    def _on_ov_done(self, job):
        if job == self.overlay_job:
            self._log(f"overlay done ({self.ov_ds.currentText()} acc) — "
                      f"geometric field beneath is data-free")

    # ---------------- gallery
    def open_gallery(self):
        if self.frame is None: return
        dlg = QDialog(self); dlg.setWindowTitle("GPS-θ field gallery (weight-only)")
        v = QVBoxLayout(dlg)
        fig = Figure(figsize=(13, 8), dpi=90)
        cv = FigureCanvasQTAgg(fig)
        v.addWidget(cv)
        lo1, hi1, lo2, hi2 = self.extent
        S, T = np.meshgrid(np.linspace(lo1, hi1, 61), np.linspace(lo2, hi2, 61))
        c, _ = self._cur_ch()
        for i, kind in enumerate(FIELDS):
            axg = fig.add_subplot(2, 3, i + 1)
            G, cmap, clim, title = self._compute_field(kind, S, T)
            im = axg.imshow(G, origin="lower", extent=list(self.extent),
                            aspect="auto", cmap=cmap, interpolation="nearest")
            if clim: im.set_clim(*clim)
            axg.set_title(title, fontsize=7)
            axg.scatter(self.frame["anc_c"][:, 0], self.frame["anc_c"][:, 1],
                        marker="^", s=30, c="red", zorder=3)
            axg.scatter(self.frame["net_c"][:, 0], self.frame["net_c"][:, 1],
                        s=8, c="w", zorder=3)
            axg.plot([c[0]], [c[1]], "w+", ms=10, mew=1.6, zorder=4)
            axg.tick_params(labelsize=5)
        fig.tight_layout()
        dlg.show()
        self._gallery = dlg          # keep reference

    # ---------------- interactions
    def on_chart_click(self, ev):
        if not self.ready or ev.inaxes is not self.ax or ev.xdata is None: return
        x, y = float(ev.xdata), float(ev.ydata)
        f = self.frame
        span = max(np.ptp(self.extent[:2]), np.ptp(self.extent[2:]))
        dn = np.hypot(f["net_c"][:, 0]-x, f["net_c"][:, 1]-y)
        if dn.min() < .03*span:
            self.dest_combo.setCurrentText(self.names[int(dn.argmin())])
            self.navigate(); return
        da = np.hypot(f["anc_c"][:, 0]-x, f["anc_c"][:, 1]-y)
        if da.min() < .03*span:
            c, _, _, h, _ = self._fix(self.fleet_vecs[f["sats"][int(da.argmin())]])
            for s, v in zip(self.pos_sliders[:-1], c): s.set_value(float(v))
            self.el_slider.set_value(float(h))
        else:
            self.pos_sliders[0].set_value(float(np.clip(x, *self.extent[:2])))
            if len(self.pos_sliders) > 2:
                self.pos_sliders[1].set_value(float(np.clip(y, *self.extent[2:])))
        # near a snapshot?
        for sn in self.snaps:
            if np.hypot(sn["x"]-x, sn["y"]-y) < .025*span:
                self._goto_snap(sn); break
        self.schedule_live()

    def on_morph_click(self, ev):
        if ev.inaxes not in self.waxes or ev.xdata is None: return
        k = self.waxes.index(ev.inaxes)
        J = self.tsl[k * 2]; sh = self.SHAPES[k * 2]
        r = int(np.clip(round(ev.ydata), 0, sh[0]-1))
        ccol = int(np.clip(round(ev.xdata), 0, sh[1]-1))
        jflat = J.start + r * sh[1] + ccol
        self.scomp.setValue(jflat)
        self.strace_combo.setCurrentText("component")
        self.statusBar().showMessage(
            f"plotted θ_{jflat} (tensor {self.tlabels[k*2]}[{r},{ccol}])", 4000)

    # ---------------- section (1-D weight slice)
    def _update_section(self):
        if not self.ready or self.frame is None or not self.pos_sliders: return
        f = self.frame
        a = self.ax_combo.currentIndex(); K1 = len(f["E"])
        c, h = self._cur_ch()
        if a < K1:
            lo, hi = float(f["lo"][a]), float(f["hi"][a])
            s_vals = np.linspace(lo, hi, 161)
            base = f["mean"] + sum(ci * f["E"][i] for i, ci in enumerate(c) if i != a) \
                   + h * (f["n"] or 0.0)
            TH = base[None, :] + s_vals[:, None] * f["E"][a]
            cur = c[a]
        else:
            lo, hi = -self.el_range, self.el_range
            s_vals = np.linspace(lo, hi, 161)
            base = f["mean"] + c @ f["E"]
            TH = base[None, :] + s_vals[:, None] * f["n"]
            cur = h
        self.sax.clear()
        trace = self.strace_combo.currentText()
        if trace == "component":
            j = self.scomp.value()
            self.sax.plot(s_vals, TH[:, j], lw=1.4, color="#3b82f6",
                          label=f"θ_{j} ({self._tensor_of(j)})")
            if a < K1:
                anc_c = f["anc_c"][:, a]
                self.sax.scatter(anc_c, [self.fleet_vecs[n][j] for n in f["sats"]],
                                 marker="^", s=40, c="red", zorder=5)
                for n, x, v in zip(f["sats"], anc_c,
                                   [self.fleet_vecs[n][j] for n in f["sats"]]):
                    self.sax.annotate(n, (x, v), fontsize=5, color="#b91c1c",
                                      xytext=(3, 3), textcoords="offset points")
            self.sax.set_ylabel(f"θ_{j}")
        elif trace == "tensor norms":
            for ti, J in enumerate(self.tsl):
                self.sax.plot(s_vals, np.linalg.norm(TH[:, J], axis=1), lw=1.1,
                              label=self.tlabels[ti])
            self.sax.set_ylabel("‖θ_(j)‖")
        elif trace == "pseudoranges":
            for n in f["sats"]:
                self.sax.plot(s_vals, np.linalg.norm(TH - self.fleet_vecs[n], axis=1),
                              lw=1.8 if n in CORE_SATS else .9, label=n)
            self.sax.set_ylabel("ρᵢ(θ(s))")
        else:
            self.sax.plot(s_vals, np.linalg.norm(TH, axis=1), lw=1.4, color="#111")
            self.sax.set_ylabel("‖θ(s)‖")
        self.sax.axvline(cur, color="#f59e0b", lw=1.2)
        self.sax.set_xlabel(f"{self.ax_combo.currentText()} sweep (others fixed)")
        self.sax.set_title("1-D weight section θ(s)", fontsize=8)
        self.sax.legend(fontsize=5, ncol=2, loc="best")
        self.scanvas.draw_idle()

    def _tensor_of(self, j):
        for ti, J in enumerate(self.tsl):
            if J.start <= j < J.stop: return self.tlabels[ti]
        return "?"

    def _toggle_sweep(self):
        if self.sweep_timer.isActive():
            self.sweep_timer.stop(); self.btn_sweep.setText("sweep ▶")
        else:
            self.sweep_timer.start(); self.btn_sweep.setText("sweep ❚❚")

    def _sweep_tick(self):
        a = self.ax_combo.currentIndex()
        sl = self.pos_sliders[a] if a < len(self.pos_sliders) else None
        if sl is None: return
        v = sl.value() + self.sweep_dir * 0.012 * (sl.hi - sl.lo)
        if v >= sl.hi or v <= sl.lo:
            self.sweep_dir *= -1; v = float(np.clip(v, sl.lo, sl.hi))
        sl.set_value(v, silent=False)

    # ---------------- transit
    def _axis_label(self, i):
        return "elev" if i == len(self.frame["E"]) else f"c{i+1}"

    def navigate(self):
        self.dest = self.dest_combo.currentText()
        c_dest, eps, _, h_dest, _ = self._fix(self.core.nets[self.dest])
        c, h = self._cur_ch()
        self.dest_err = 0.0 if self.el_combo.currentText().startswith("→ dest") else eps
        self.theta_ref = self._cur_vec()
        shape = self.shape_combo.currentText()
        targets = list(c_dest) + [h_dest]
        cur = list(c) + [h]
        def leg_moves(frm, to):
            return [dict(axis=i, frm=float(frm[i]), to=float(to[i]),
                         d=abs(float(to[i]-frm[i])))
                    for i in range(len(frm)) if abs(to[i]-frm[i]) > 1e-9]
        if shape == "via anchor":
            ac = self.frame["anc_c"]
            j = int(np.linalg.norm(ac - c, axis=1).argmin())
            anc_vec = self.fleet_vecs[self.frame["sats"][j]]
            c_a, _, _, h_a, _ = self._fix(anc_vec)
            self.route = leg_moves(cur, list(c_a) + [h_a]) + \
                         [dict(via=self.frame["sats"][j])] + \
                         leg_moves(list(c_a) + [h_a], targets)
        elif shape == "direct":
            self.route = [dict(direct=True,
                               frms=np.array(cur), tos=np.array(targets),
                               total=float(np.linalg.norm(np.array(targets)-np.array(cur))),
                               p=0.0)]
        else:
            self.route = leg_moves(cur, targets)
        self.total_len = float(np.linalg.norm(np.array(targets) - np.array(cur)))
        self.step_i = 0
        self._render_route(); self._render_steps()
        self._log(f"navigate → {self.dest} [{shape}]: {self.total_len:.1f} u, "
                  f"{'exact landing (elevator)' if self.dest_err == 0 else f'ε={eps:.2f}'}")
        if not self.driving: self._toggle_drive()

    def cancel_route(self):
        self.dest, self.route = None, None
        if self.driving: self._toggle_drive()
        self.route_line.set_data([], []); self.dest_mark.set_data([], [])
        self.steps.clear(); self._do_live()

    def _toggle_drive(self):
        if not self.route: return
        self.driving = not self.driving
        self.btn_drive.setText("Stop" if self.driving else "Drive")
        self.drive_timer.start() if self.driving else self.drive_timer.stop()

    def _drive_tick(self):
        if not self.route or self.step_i >= len(self.route):
            if self.driving: self._toggle_drive()
            vec = self._cur_vec()
            dd = float(np.linalg.norm(vec - self.core.nets[self.dest])) if self.dest else 0
            self.statusBar().showMessage(
                f"🏁 landed at {self.dest} · final ‖θ−dest‖={dd:.4f}", 8000)
            self._log(f"arrived at {self.dest} · ‖θ−dest‖={dd:.4f}")
            self._render_steps(); return
        st = self.route[self.step_i]
        if "via" in st:
            self.statusBar().showMessage(f"checkpoint ▲ {st['via']}", 2500)
            self.step_i += 1; self._render_steps(); return
        if st.get("direct"):
            dt = self.drive_timer.interval() / 1000.0
            st["p"] = min(1.0, st["p"] + self.speed.value() * dt / max(st["total"], 1e-9))
            vals = st["frms"] + st["p"] * (st["tos"] - st["frms"])
            for sl, v in zip(self.pos_sliders, vals): sl.set_value(float(v))
            if st["p"] >= 1.0: self.step_i += 1
        else:
            sl = self.pos_sliders[st["axis"]]
            stepv = np.sign(st["to"]-sl.value()) * self.speed.value() * \
                    self.drive_timer.interval()/1000.0
            if abs(stepv) >= abs(st["to"]-sl.value()):
                sl.set_value(st["to"]); self.step_i += 1
            else:
                sl.set_value(sl.value()+stepv)
        self._render_steps(); self.schedule_live()

    def _progress(self):
        if not self.route: return 0.0, 0.0
        done = 0.0
        for st in self.route:
            if "via" in st: continue
            if st.get("direct"):
                done += st["total"] * st["p"]
            elif self.route.index(st) < self.step_i:
                done += st["d"]
            elif self.route.index(st) == self.step_i:
                sl = self.pos_sliders[st["axis"]]
                done += abs(sl.value() - st["frm"])
        return done/max(self.total_len, 1e-9), max(self.total_len-done, 0.0)

    def _render_route(self):
        if not self.route: self.route_line.set_data([], []); return
        c, _ = self._cur_ch(); cur = c.copy(); pts = [(cur[0], cur[1])]
        for st in self.route:
            if "via" in st: continue
            if st.get("direct"):
                pts.append((st["tos"][0], st["tos"][1]))
            elif st["axis"] < 2:
                cur[st["axis"]] = st["to"]; pts.append((cur[0], cur[1]))
        self.route_line.set_data([p[0] for p in pts], [p[1] for p in pts])
        if self.dest:
            cc = self.frame["net_c"][self.names.index(self.dest)]
            self.dest_mark.set_data([cc[0]], [cc[1]])

    def _render_steps(self):
        if not self.route: self.steps.clear(); return
        lines = [f"ROUTE → {self.dest} ({self.total_len:.1f} u, "
                 f"{self.shape_combo.currentText()})"]
        for i, st in enumerate(self.route):
            if "via" in st:
                lines.append(f"    ⚑ VIA ▲ {st['via']}"); continue
            if st.get("direct"):
                lines.append(f"{'▶' if self.driving else ' '} direct · all axes · "
                             f"{st['p']:5.1%}")
                continue
            m = "▶" if (i == self.step_i and self.driving) else ("✓" if i < self.step_i else " ")
            lines.append(f"{m} {i+1:2d}. {self._axis_label(st['axis']):4s} "
                         f"{st['frm']:+8.2f} → {st['to']:+8.2f}  ({st['d']:.2f})")
        if self.step_i >= len(self.route):
            lines.append(f"🏁 landed · err={self.dest_err:.4f}")
        self.steps.setPlainText("\n".join(lines))

    # ---------------- snapshots
    def add_snapshot(self):
        if not self.ready: return
        c, h = self._cur_ch()
        self.snaps.append(dict(x=float(c[0]), y=float(c[1]),
                               vals=[s.value() for s in self.pos_sliders],
                               n=len(self.snaps)+1))
        self._render_snaps(); self._log(f"snapshot S{len(self.snaps)} stored")

    def del_snapshot(self):
        if self.snaps:
            self.snaps.pop(); self._render_snaps()

    def _render_snaps(self):
        for a in self.snap_artists:
            try: a.remove()
            except Exception: pass
        self.snap_artists = []
        for sn in self.snaps:
            ln, = self.ax.plot([sn["x"]], [sn["y"]], "o", ms=9, mfc="none",
                               mec="cyan", mew=1.5, zorder=7)
            tx = self.ax.annotate(f"S{sn['n']}", (sn["x"], sn["y"]), color="cyan",
                                  fontsize=6, zorder=7, xytext=(4, -2),
                                  textcoords="offset points")
            self.snap_artists += [ln, tx]
        self.canvas.draw_idle()

    def _goto_snap(self, sn):
        for sl, v in zip(self.pos_sliders, sn["vals"]): sl.set_value(v)
        self.schedule_live()

    # ---------------- live
    def schedule_live(self): self.live_timer.start()

    def _on_user_move(self, _=0.0):
        self.schedule_live()
        self._update_section()
        if self.route: self.recalc_timer.start()

    def _do_live(self):
        if not self.ready or self.frame is None or not self.pos_sliders: return
        c, h = self._cur_ch()
        vec = self.vec_of(c, h)
        c_hat, eps, pr, h_hat, res = self._fix(vec)
        f = self.frame
        lines = [f"FIX {len(f['E'])}-D+elev · {len(f['sats'])} anchors · κ={f['cond']:.0f}",
                 f"ε={eps:.2e}  h={h:+.2f}  solver|r|={res:.1e}",
                 "POS " + " ".join(f"c{i+1}={v:+.1f}" for i, v in enumerate(c[:5]))
                 + ("…" if len(c) > 5 else "")]
        if self.dest:
            p = self._progress()
            eta = p[1]/max(self.speed.value(), 1e-9) if self.driving else None
            lines.append(f"DST {self.dest}  left={p[1]:.2f} ({p[0]:5.1%})"
                         + (f"  eta~{eta:.0f}s" if eta is not None else ""))
        d = np.linalg.norm(self.M - vec, axis=1); kn = int(d.argmin())
        lines.append(f"NEAR {self.names[kn]} (d={d[kn]:.2f})")
        self.info.setPlainText("\n".join(lines))

        order = np.argsort(pr); dmax = max(pr[order[-1]], 1e-9)
        sig = [f"{'▲' if f['sats'][i] in CORE_SATS else '·'} "
               f"{f['sats'][i]:12s} {pr[i]:8.2f}  "
               f"{'█'*max(1, int(round(10*(1-pr[i]/dmax))))}" for i in order]
        self.signal.setPlainText("\n".join(sig))

        lam = self._composition(c)
        self.compo.set_data(f["sats"], lam,
                            f"composition (Σλ={lam.sum():.3f}) — signed: mix of satellites")
        ref = self.theta_ref if self.theta_ref is not None else f["mean"]
        dvec = vec - ref
        shares = np.array([np.linalg.norm(dvec[J]) for J in self.tsl])
        tot = shares.sum()
        self.attrib.set_data(self.tlabels, shares/tot if tot > 1e-9 else shares*0,
                             f"attribution vs {'route start' if self.route else 'ref'} "
                             f"(‖Δ‖={tot:.2f})")

        self.dot.set_data([c[0]], [c[1]])
        self.htext.set_text(f"elev h={h:+.2f}" if abs(h) > 1e-9 else "")
        self.htext.set_color("#f59e0b" if abs(h) > 1e-9 else "none")
        self._render_route()
        self._update_viewer()
        self.canvas.draw_idle()

    def _update_viewer(self):
        if not self.ready or self.frame is None: return
        th = self._cur_vec()
        mode = self.wview.currentText()
        ref = None
        if mode == "θ − mean":                 ref = self.frame["mean"]
        elif mode == "θ − nearest net":
            ref = self.M[int(np.linalg.norm(self.M - th, axis=1).argmin())]
        elif mode == "θ − dest" and self.dest: ref = self.core.nets[self.dest]
        V = th if ref is None else th - ref
        for k, (im, J) in enumerate(zip(self.wims[:3], (0, 2, 4))):
            W = V[self.tsl[J]].reshape(self.SHAPES[J])
            mx = max(np.abs(W).max(), 1e-6)
            im.set_data(W); im.set_clim(-mx, mx)
            self.waxes[k].set_xlabel(f"‖{self.tlabels[J]}‖={np.linalg.norm(W):.2f}",
                                     fontsize=6)
        b = np.concatenate([V[self.tsl[j]] for j in (1, 3, 5)])[None, :]
        self.wims[3].set_data(b)
        mx = max(np.abs(b).max(), 1e-6)
        self.wims[3].set_clim(-mx, mx)
        self.wcanvas.draw_idle()

    def snap_nearest(self):
        d = np.linalg.norm(self.M - self._cur_vec(), axis=1)
        c, _, _, h, _ = self._fix(self.M[int(d.argmin())])
        for s, v in zip(self.pos_sliders[:-1], c): s.set_value(float(v))
        self.el_slider.set_value(float(h))       # exact landing via elevator
        self.schedule_live()

    # ---------------- studies (weight-only)
    def study_coverage(self):
        names_all = self.fleet_order; rng = np.random.RandomState(0)
        lines = ["COVERAGE — anchor span vs between-net variance",
                 " K   mean ε   median ε   R²"]
        total_var = float(self.M.var(0).sum())
        for k in range(4, len(names_all)+1):
            alts, r2s = [], []
            for _ in range(6 if k < len(names_all) else 1):
                sub = rng.choice(len(names_all), k, replace=False)
                X = np.stack([self.fleet_vecs[names_all[i]] for i in sub])
                Xc = X - X.mean(0); kk = k-1
                if np.linalg.matrix_rank(Xc) < kk: continue
                _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
                E = Vt[:kk]; mean = X.mean(0)
                cc = (self.M - mean) @ E.T
                alts.append(np.linalg.norm(self.M - (mean + cc @ E), axis=1))
                r2s.append(float(cc.var(0).sum() / max(total_var, 1e-12)))
            if not alts: continue
            alt = np.concatenate(alts)
            lines.append(f"{k:3d}  {alt.mean():8.2f}  {np.median(alt):8.2f}   {np.mean(r2s):8.1%}")
        self._log("coverage study:\n" + "\n".join(lines))
        self.statusBar().showMessage("coverage study → journal", 4000)

    def study_gdop(self):
        names_all = self.fleet_order; rng = np.random.RandomState(1)
        results = []
        for _ in range(300):
            sub = rng.choice(len(names_all), len(self.active), replace=False)
            X = np.stack([self.fleet_vecs[names_all[i]] for i in sub])
            Xc = X - X.mean(0)
            if np.linalg.matrix_rank(Xc) < len(self.active)-1: continue
            S = np.linalg.svd(Xc, compute_uv=False)
            results.append((S[0]/S[-1], tuple(names_all[i] for i in sub)))
        results.sort()
        lines = [f"GDOP — κ of random {len(self.active)}-anchor constellations"]
        for κ, sub in results[:3]:  lines.append(f"best  κ={κ:10.1f}  {', '.join(sub)}")
        for κ, sub in results[-3:]: lines.append(f"worst κ={κ:10.1f}  {', '.join(sub)}")
        self._log("GDOP study:\n" + "\n".join(lines))
        self.statusBar().showMessage("GDOP study → journal", 4000)

    def study_geometry(self):
        f = self.frame
        A = f["B"] - f["mean"]
        An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-300)
        C = An @ An.T
        pairs = sorted(((C[i, j], f["sats"][i], f["sats"][j])
                        for i in range(len(A)) for j in range(i+1, len(A))))
        lines = [f"GEOMETRY — anchor pairwise angles (κ={f['cond']:.0f})"]
        for cs, a, b in pairs[:4]:
            lines.append(f"  {a:12s} ↔ {b:12s}  {np.degrees(np.arccos(np.clip(cs,-1,1))):7.1f}°")
        lines.append("  (small angle ⇒ near-collinear ⇒ GDOP penalty)")
        self._log("constellation geometry:\n" + "\n".join(lines))
        self.statusBar().showMessage("geometry study → journal", 4000)

    def copy_methods(self):
        f = self.frame
        para = (
            f"Synthetic anchor constellation with elevator. K={len(f['sats'])} "
            f"training-free anchors in R^N (N={self.N_PARAMS:,}): "
            f"{', '.join(f['sats'])}. SVD of the centered anchor matrix yields an "
            f"orthonormal chart (mean, E in R^({len(f['E'])}xN)); a unit vector n "
            f"orthogonal to span(E) serves as the elevator axis. Position: "
            f"theta = mean + c@E + h*n. Localization solves the differenced "
            f"pseudorange equations 2(a1-ai)^T theta = rho_i^2 - rho_1^2 - ||a_i||^2 "
            f"+ ||a_1||^2 (max residual reported); the elevator coordinate h absorbs "
            f"the off-span component, so trained networks are exactly reachable. "
            f"Barycentric composition lambda solves lambda@D = c@E, sum(lambda)=1. "
            f"Charts are closed-form weight fields (pseudoranges, range differences, "
            f"norms, cosines, Voronoi cells, per-tensor norms); accuracy was used "
            f"only as an optional overlay. GDOP kappa={f['cond']:.0f}.")
        QApplication.clipboard().setText(para)
        self._log("methods paragraph copied")
        self.statusBar().showMessage("methods ¶ copied", 4000)

    def export_journal(self):
        from PySide6.QtWidgets import QFileDialog
        fn, _ = QFileDialog.getSaveFileName(self, "Export journal", "gps_theta_journal.txt",
                                            "Text (*.txt)")
        if fn:
            with open(fn, "w") as fh: fh.write(self.journal.toPlainText())
            self.statusBar().showMessage(f"journal → {fn}", 4000)

    def save_png(self):
        from PySide6.QtWidgets import QFileDialog
        fn, _ = QFileDialog.getSaveFileName(self, "Save PNG", "gps_theta_v8.png", "PNG (*.png)")
        if fn: self.fig.savefig(fn, dpi=160)

    def closeEvent(self, ev):
        self.sweep_timer.stop(); self.drive_timer.stop()
        self.live_timer.stop(); self.recalc_timer.stop()
        if self.overlay_worker and self.overlay_worker.isRunning():
            self.overlay_worker.stop = True; self.overlay_worker.wait(2000)
        super().closeEvent(ev)

# ================================================================ entry
def main():
    app = QApplication(sys.argv); app.setStyle("Fusion")
    print(f"device: {DEVICE}")
    core = Core(); ensure_nets(core)
    win = MainWindow(core); win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()