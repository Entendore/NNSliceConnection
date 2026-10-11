"""
Weight-space explorer v4 — PySide6 desktop GUI.

16 equal-size networks (3,466 params each) as points in R^N:
  spec:<ds>   6 specialists (seed 0)      spec2:<ds>  6 variants (seed 1000)
  gen:ALL     generalist                  const:0.5/0.1/0.9  constant baselines

Modes: PLANE(s,t) · VOLUME(s,t,u) · LINE(m, A→B) · BASIN(s,t around a net) ·
MIX(16 knobs, one per net). All heavy work (training, heatmap) in QThreads;
the UI stays responsive; the heatmap streams in row by row.
"""

import os
os.environ.setdefault("QT_API", "PySide6")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import time
from threading import Lock
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from PySide6.QtCore import Qt, QThread, Signal, QTimer, QEventLoop, QRectF, QPointF
from PySide6.QtGui import (QPainter, QColor, QPen, QImage, QPixmap, QFont,
                           QAction, QShortcut, QKeySequence)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QSlider, QComboBox,
    QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QRadioButton, QButtonGroup, QStackedWidget, QTextEdit, QSpinBox,
    QProgressDialog, QFileDialog, QSplitter, QFrame,
)

import matplotlib as mpl
mpl.use("qtagg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# ================================================================ config
SEED, MAX_F, MAX_C = 0, 64, 10
HIDDEN, EPOCHS, BATCH, LR = (32, 32), 300, 32, 1e-3
CACHE     = "net_vectors_v4.npz"        # delete to force retraining
CONSTANTS = (0.5, 0.1, 0.9)
METRICS   = ("acc", "conf", "loss")
MODES     = ["PLANE", "VOLUME", "LINE", "BASIN", "MIX"]
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# ================================================================ ML core
def to_class_ids(y):
    y = np.asarray(y, float).ravel()
    if len(np.unique(y)) > MAX_C:
        edges = np.quantile(y, np.linspace(0, 1, MAX_C + 1)[1:-1])
        y = np.searchsorted(edges, y, side="right")
    return np.unique(y, return_inverse=True)[1].astype(np.int64)

def get_data():
    loaders = {
        "iris":   datasets.load_iris,          "wine":   datasets.load_wine,
        "cancer": datasets.load_breast_cancer, "digits": datasets.load_digits,
        "diab":   datasets.load_diabetes,      "linn":   datasets.load_linnerud,
    }
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

def constant_model(c):
    m = make_model()
    with torch.no_grad():
        for p in m.parameters(): p.fill_(c)
    return m

def flat_of(m):
    return torch.cat([p.data.reshape(-1) for p in m.parameters()]
                     ).cpu().numpy().astype(np.float64)

def set_flat(m, vec):
    i = 0
    with torch.no_grad():
        for p in m.parameters():
            n = p.numel()
            p.copy_(torch.from_numpy(vec[i:i+n].astype(np.float32)).reshape_as(p)); i += n

def filter_normalize(dir_vec, ref_vec, model):
    out, i = dir_vec.copy(), 0
    for p in model.parameters():
        n = p.numel()
        out[i:i+n] *= (ref_vec[i:i+n].std() + 1e-12) / (out[i:i+n].std() + 1e-12)
        i += n
    return out

class Core:
    """Data + nets + fast metrics. All model access goes through one lock so
    the UI thread and worker threads can share the same nn.Module safely."""
    def __init__(self):
        self.lock = Lock()
        self.data = get_data()
        self.ds_list = list(self.data)
        self.model = make_model()
        self.Xbig = torch.cat([torch.from_numpy(self.data[d][2]) for d in self.ds_list]).to(DEVICE)
        self.ybig = torch.cat([torch.from_numpy(self.data[d][3]) for d in self.ds_list]).to(DEVICE)
        self.bounds = np.cumsum([0] + [len(self.data[d][2]) for d in self.ds_list])
        self.nets = None

    def load_from_cache(self):
        z = np.load(CACHE)
        self.nets = {k: z[k] for k in z.files}

    def train_all(self, progress=None):
        nets, tasks = {}, []
        for i, name in enumerate(self.ds_list):
            Xtr, ytr = self.data[name][0], self.data[name][1]
            tasks += [(f"spec:{name}",  Xtr, ytr, SEED + i * 10),
                      (f"spec2:{name}", Xtr, ytr, SEED + 1000 + i * 10)]
        Xa = np.concatenate([self.data[d][0] for d in self.ds_list])
        ya = np.concatenate([self.data[d][1] for d in self.ds_list])
        tasks.append(("gen:ALL", Xa, ya, SEED))
        for k, (name, X, y, sd) in enumerate(tasks):
            if progress: progress(k, len(tasks), name)
            torch.manual_seed(sd)
            nets[name] = flat_of(train(make_model(), X, y))
        for c in CONSTANTS:
            nets[f"const:{c}"] = flat_of(constant_model(c))
        np.savez(CACHE, **nets)
        self.nets = nets
        if progress: progress(len(tasks), len(tasks), "done")

    def metrics(self, vec):
        with self.lock:
            set_flat(self.model, vec)
            with torch.no_grad():
                logits = self.model(self.Xbig)
            out = {}
            for i, d in enumerate(self.ds_list):
                lg = logits[self.bounds[i]:self.bounds[i+1]]
                yy = self.ybig[self.bounds[i]:self.bounds[i+1]]
                out[d] = ((lg.argmax(1) == yy).float().mean().item(),
                          lg.softmax(1).max(1).values.mean().item(),
                          F.cross_entropy(lg, yy).item())
        out["ALL"] = tuple(np.mean([out[d][j] for d in self.ds_list]) for j in range(3))
        return out

    def sample_probs(self, vec, ds, idx):
        i0 = self.ds_list.index(ds)
        with self.lock:
            set_flat(self.model, vec)
            with torch.no_grad():
                sl = slice(self.bounds[i0] + idx, self.bounds[i0] + idx + 1)
                return self.model(self.Xbig[sl]).softmax(1)[0].cpu().numpy()

    def sample(self, ds, idx):
        i0 = self.ds_list.index(ds)
        n = self.bounds[i0+1] - self.bounds[i0]
        idx = int(np.clip(idx, 0, n - 1))
        return (self.Xbig[self.bounds[i0]+idx].cpu().numpy(),
                int(self.ybig[self.bounds[i0]+idx]))

# ================================================================ workers
class TrainWorker(QThread):
    progress = Signal(int, int, str)
    done = Signal()
    def __init__(self, core):
        super().__init__(); self.core = core
    def run(self):
        self.core.train_all(lambda i, n, name: self.progress.emit(i, n, name))
        self.done.emit()

class MapWorker(QThread):
    row_ready = Signal(int, int, np.ndarray, float, float)  # job, row, values, vmin, vmax
    done = Signal(int, bool)
    def __init__(self, job, eval_fn, origin, d1, d2, extent, res):
        super().__init__()
        self.job, self.eval_fn = job, eval_fn
        self.origin, self.d1, self.d2, self.extent, self.res = origin, d1, d2, extent, res
        self.stop = False
    def run(self):
        lo1, hi1, lo2, hi2 = self.extent
        xs = np.linspace(lo1, hi1, self.res); ys = np.linspace(lo2, hi2, self.res)
        vmin = vmax = None; completed = True
        for r, y in enumerate(ys):
            if self.stop: completed = False; break
            row = np.array([self.eval_fn(self.origin + x*self.d1 + y*self.d2) for x in xs])
            vmin = float(row.min()) if vmin is None else min(vmin, float(row.min()))
            vmax = float(row.max()) if vmax is None else max(vmax, float(row.max()))
            self.row_ready.emit(self.job, r, row, vmin, vmax)
        self.done.emit(self.job, completed)

def ensure_nets(core):
    """Train on first launch behind a modal progress dialog; otherwise load cache."""
    if os.path.exists(CACHE):
        core.load_from_cache(); return
    dlg = QProgressDialog("Preparing networks…", None, 0, 1)
    dlg.setWindowModality(Qt.ApplicationModal); dlg.setMinimumWidth(380); dlg.show()
    w = TrainWorker(core)
    w.progress.connect(lambda i, n, name: (dlg.setMaximum(n), dlg.setValue(i),
                                           dlg.setLabelText(f"training {name} …")))
    w.start()
    if not w.isFinished():
        loop = QEventLoop(); w.done.connect(loop.quit); loop.exec()
    w.wait(); dlg.close(); dlg.deleteLater()

# ================================================================ custom widgets
class LabeledSlider(QWidget):
    valueChangedF = Signal(float)
    def __init__(self, text, lo, hi, val, dec=3, parent=None):
        super().__init__(parent)
        self.dec, self.scale, self.lo, self.hi = dec, 10 ** dec, lo, hi
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.lab = QLabel(text); self.lab.setMinimumWidth(90)
        self.sl = QSlider(Qt.Horizontal)
        self.sl.setRange(int(lo*self.scale), int(hi*self.scale))
        self.sl.setValue(int(val*self.scale))
        self.val_lab = QLabel(f"{val:.{dec}f}"); self.val_lab.setMinimumWidth(62)
        f = QFont("monospace", 8); self.val_lab.setFont(f)
        lay.addWidget(self.lab); lay.addWidget(self.sl, 1); lay.addWidget(self.val_lab)
        self.sl.valueChanged.connect(self._emit)
    def _emit(self, v):
        f = v / self.scale
        self.val_lab.setText(f"{f:.{self.dec}f}")
        self.valueChangedF.emit(f)
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

class BarChart(QWidget):
    def __init__(self, labels, color="#3b82f6", parent=None):
        super().__init__(parent)
        self.labels, self.color = labels, color
        self.vals, self.ymax, self.title, self.hi = np.zeros(len(labels)), 1.0, "", -1
        self.setMinimumHeight(150)
    def set_data(self, vals, ymax, title, hi=-1):
        self.vals, self.ymax, self.title, self.hi = np.asarray(vals), ymax, title, hi
        self.update()
    def paintEvent(self, ev):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        left, right, top, bottom = 6, 6, 18, 30
        pw, ph = w - left - right, h - top - bottom
        n = len(self.vals); bw = pw / max(n, 1)
        p.setPen(QColor("#666"))
        p.drawText(QRectF(0, 0, w, top), Qt.AlignCenter, self.title)
        p.drawLine(QPointF(left, h - bottom), QPointF(w - right, h - bottom))
        for i, v in enumerate(self.vals):
            bh = ph * float(np.clip(v / max(self.ymax, 1e-9), 0, 1))
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#f59e0b") if i == self.hi else QColor(self.color))
            p.drawRect(QRectF(left + i*bw + 2, h - bottom - bh, max(bw - 4, 1), bh))
            p.setPen(QColor("#999"))
            p.drawText(QRectF(left + i*bw, h - bottom - bh - 12, bw, 12),
                       Qt.AlignCenter, f"{v:.2f}")
            p.save()
            p.translate(left + i*bw + bw/2, h - bottom + 2); p.rotate(-40)
            p.drawText(0, 10, self.labels[i]); p.restore()

class ProbBars(QWidget):
    def __init__(self, n, parent=None):
        super().__init__(parent)
        self.n, self.vals, self.true = n, np.zeros(n), -1
        self.setMinimumHeight(120)
    def set_data(self, vals, true):
        self.vals, self.true = np.asarray(vals), true; self.update()
    def paintEvent(self, ev):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        rowh = (h - 4) / self.n
        mx = max(float(self.vals.max()), 1e-6)
        for i in range(self.n):
            y = 2 + i * rowh
            p.setPen(QPen(QColor("#ffd54f") if i == self.true else QColor("#777"), 1))
            p.drawText(QRectF(0, y, 18, rowh), Qt.AlignVCenter | Qt.AlignRight, str(i))
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#ff7043") if i == self.true else QColor("#ffcc80"))
            p.drawRect(QRectF(24, y + rowh*0.15, (w - 62) * self.vals[i] / mx, rowh*0.7))
            p.setPen(QColor("#aaa"))
            p.drawText(QRectF(w - 34, y, 32, rowh), Qt.AlignVCenter, f"{self.vals[i]:.2f}")

# ================================================================ main window
class MainWindow(QMainWindow):
    def __init__(self, core: Core):
        super().__init__()
        self.core = core
        self.names = list(core.nets)
        self.M = np.stack([core.nets[n] for n in self.names])
        self.N_PARAMS = self.M.shape[1]
        pca = PCA(3).fit(self.M)
        self.mu = pca.mean_
        self.U1, self.U2, self.U3 = pca.components_
        proj = pca.transform(self.M)
        self.P2, self.P3 = proj[:, :2], proj[:, 2]
        print(f"PC1-3 capture {pca.explained_variance_ratio_.sum():.0%} of between-net variance")
        span = np.maximum(np.ptp(self.P2, axis=0), 1e-9)
        self.plane_extent = (self.P2[:, 0].min() - .2*span[0], self.P2[:, 0].max() + .2*span[0],
                             self.P2[:, 1].min() - .2*span[1], self.P2[:, 1].max() + .2*span[1])
        self.u_range = float(1.5 * np.abs(self.P3).max() + 1e-9)
        self.trained = [n for n in self.names if not n.startswith("const:")]

        self.basin, self.basin_seed = None, 1
        self.waypoints, self.wp_artists = [], []
        self.trail = deque(maxlen=500)
        self.map_cache, self.job, self.map_worker = {}, 0, None
        self.playing, self.play_dir = False, 1
        self._img_key, self.ready = None, False

        self._build_ui()
        self._fill_net_widgets()
        self._wire()
        self._draw_static_map()
        self.ready = True
        self.set_mode(0)

    # ---------------- UI construction
    def _build_ui(self):
        self.setWindowTitle(f"Weight-space explorer — {self.N_PARAMS:,} params/net, "
                            f"{len(self.names)} nets")
        self.resize(1500, 920)

        mb = self.menuBar()
        m = mb.addMenu("&File")
        for text, slot in (("Save &PNG…", self.save_png),
                           ("Save &weight vector…", self.save_vec)):
            a = QAction(text, self); a.triggered.connect(slot); m.addAction(a)
        m.addSeparator()
        aq = QAction("&Quit", self); aq.triggered.connect(self.close); m.addAction(aq)

        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central); root.setContentsMargins(4, 4, 4, 4)
        splitter = QSplitter(Qt.Vertical); root.addWidget(splitter, 1)

        # ---- top: map + right panel
        top = QWidget(); th = QHBoxLayout(top); th.setContentsMargins(2, 2, 2, 2)
        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111)
        th.addWidget(self.canvas, 2)

        right = QWidget(); rv = QVBoxLayout(right)
        self.bars = BarChart(self.core.ds_list)
        self.info = QTextEdit(); self.info.setReadOnly(True)
        self.info.setFont(QFont("monospace", 8)); self.info.setMaximumHeight(112)
        insp = QGroupBox("Inspector — sample from the focus dataset")
        iv = QHBoxLayout(insp)
        self.img_label = QLabel(); self.img_label.setFixedSize(96, 96)
        self.img_label.setFrameShape(QFrame.Box)
        self.probs = ProbBars(MAX_C)
        iv.addWidget(self.img_label); iv.addWidget(self.probs, 1)
        samp_row = QHBoxLayout()
        samp_row.addWidget(QLabel("sample idx")); self.samp_spin = QSpinBox()
        samp_row.addWidget(self.samp_spin); samp_row.addStretch(1)
        rv.addWidget(self.bars, 3); rv.addWidget(self.info); rv.addWidget(insp)
        rv.addLayout(samp_row)
        th.addWidget(right, 1)
        splitter.addWidget(top)

        # ---- bottom: controls
        bottom = QWidget(); bh = QHBoxLayout(bottom)
        g_mode = QGroupBox("Mode"); mv = QVBoxLayout(g_mode)
        self.mode_group = QButtonGroup(self); self.mode_btns = []
        for i, name in enumerate(MODES):
            rb = QRadioButton(name); rb.setChecked(i == 0)
            rb.setFocusPolicy(Qt.NoFocus)
            self.mode_group.addButton(rb, i); mv.addWidget(rb); self.mode_btns.append(rb)
        bh.addWidget(g_mode)

        self.pages = QStackedWidget(); bh.addWidget(self.pages, 2)
        lo1, hi1, lo2, hi2 = self.plane_extent
        gi = self.names.index("gen:ALL")

        pg = QWidget(); pv = QVBoxLayout(pg); pv.setContentsMargins(2, 2, 2, 2)
        self.pl_s = LabeledSlider("s (PC1)", lo1, hi1, self.P2[gi, 0])
        self.pl_t = LabeledSlider("t (PC2)", lo2, hi2, self.P2[gi, 1])
        pv.addWidget(self.pl_s); pv.addWidget(self.pl_t)
        pv.addWidget(QLabel("click the map to jump; click near a red dot to snap"))
        pv.addStretch(1); self.pages.addWidget(pg)

        pg = QWidget(); vv = QVBoxLayout(pg); vv.setContentsMargins(2, 2, 2, 2)
        self.vo_s = LabeledSlider("s (PC1)", lo1, hi1, self.P2[gi, 0])
        self.vo_t = LabeledSlider("t (PC2)", lo2, hi2, self.P2[gi, 1])
        self.vo_u = LabeledSlider("u (PC3)", -self.u_range, self.u_range, self.P3[gi])
        for s in (self.vo_s, self.vo_t, self.vo_u): vv.addWidget(s)
        vv.addWidget(QLabel("map shows the u = 0 slice"))
        vv.addStretch(1); self.pages.addWidget(pg)

        pg = QWidget(); lv = QVBoxLayout(pg); lv.setContentsMargins(2, 2, 2, 2)
        self.li_m = LabeledSlider("morph m", 0.0, 1.0, 0.5)
        lv.addWidget(self.li_m)
        for label, attr in (("A:", "li_a"), ("B:", "li_b")):
            row = QHBoxLayout(); row.addWidget(QLabel(label))
            c = QComboBox(); setattr(self, attr, c); row.addWidget(getattr(self, attr), 1)
            lv.addLayout(row)
        lv.addWidget(QLabel("clicking the map projects onto the A→B line"))
        lv.addStretch(1); self.pages.addWidget(pg)

        pg = QWidget(); bv = QVBoxLayout(pg); bv.setContentsMargins(2, 2, 2, 2)
        self.ba_s = LabeledSlider("s (r1)", -1, 1, 0.0)
        self.ba_t = LabeledSlider("t (r2)", -1, 1, 0.0)
        bv.addWidget(self.ba_s); bv.addWidget(self.ba_t)
        row = QHBoxLayout(); row.addWidget(QLabel("center"))
        self.ba_center = QComboBox(); row.addWidget(self.ba_center, 1)
        self.btn_dirs = QPushButton("new random dirs"); self.btn_dirs.setFocusPolicy(Qt.NoFocus)
        row.addWidget(self.btn_dirs); bv.addLayout(row)
        bv.addWidget(QLabel("filter-normalized random slice through the center net"))
        bv.addStretch(1); self.pages.addWidget(pg)

        pg = QWidget(); mx = QVBoxLayout(pg); mx.setContentsMargins(2, 2, 2, 2)
        grid_holder = QWidget(); self.mix_grid = QGridLayout(grid_holder)
        self.mix_grid.setContentsMargins(0, 0, 0, 0)
        self.mix_sliders = []
        for k, name in enumerate(self.names):
            s = LabeledSlider(name, 0.0, 1.0, 0.0, dec=2)
            s.lab.setMinimumWidth(72); s.val_lab.setMinimumWidth(40)
            s.lab.setStyleSheet("font-size:7pt;"); s.val_lab.setStyleSheet("font-size:7pt;")
            self.mix_grid.addWidget(s, k % 8, k // 8)
            self.mix_sliders.append(s)
        self.mix_sliders[gi].set_value(1.0)
        mx.addWidget(grid_holder)
        presets = QHBoxLayout()
        for text, slot in (("gen only", self._preset_gen),
                           ("uniform trained", self._preset_uniform),
                           ("all ½", self._preset_half)):
            b = QPushButton(text); b.setFocusPolicy(Qt.NoFocus)
            b.clicked.connect(slot); presets.addWidget(b)
        presets.addStretch(1); mx.addLayout(presets)
        self.pages.addWidget(pg)

        g_disp = QGroupBox("Display"); dv = QVBoxLayout(g_disp)
        self.metric_group = QButtonGroup(self)
        for i, name in enumerate(("accuracy", "confidence", "loss")):
            rb = QRadioButton(name); rb.setChecked(i == 0); rb.setFocusPolicy(Qt.NoFocus)
            self.metric_group.addButton(rb, i); dv.addWidget(rb)
        dv.addWidget(QLabel("focus dataset"))
        self.focus_combo = QComboBox(); self.focus_combo.addItems(["ALL"] + self.core.ds_list)
        dv.addWidget(self.focus_combo)
        dv.addWidget(QLabel("map resolution"))
        self.res_spin = QSpinBox(); self.res_spin.setRange(9, 41)
        self.res_spin.setSingleStep(2); self.res_spin.setValue(21)
        dv.addWidget(self.res_spin)
        self.btn_remap = QPushButton("recompute map"); self.btn_remap.setFocusPolicy(Qt.NoFocus)
        dv.addWidget(self.btn_remap); dv.addStretch(1)
        bh.addWidget(g_disp)

        g_act = QGroupBox("Actions"); av = QGridLayout(g_act)
        self.btn_play = QPushButton("Play"); self.btn_play.setFocusPolicy(Qt.NoFocus)
        btns = [(self.btn_play, self._toggle_play), (QPushButton("WP +"), self.add_wp),
                (QPushButton("WP −"), self.del_wp), (QPushButton("snap"), self.snap_nearest),
                (QPushButton("PNG"), self.save_png), (QPushButton("vec"), self.save_vec)]
        for k, (b, slot) in enumerate(btns):
            b.setFocusPolicy(Qt.NoFocus); b.clicked.connect(slot)
            av.addWidget(b, k // 2, k % 2)
        bh.addWidget(g_act)
        splitter.addWidget(bottom)
        splitter.setSizes([620, 300])

        self.live_timer = QTimer(self); self.live_timer.setSingleShot(True)
        self.live_timer.setInterval(30); self.live_timer.timeout.connect(self._do_live)
        self.play_timer = QTimer(self); self.play_timer.setInterval(60)
        self.play_timer.timeout.connect(self._step)
        hint = QLabel(" 1-5 mode · Space play · R snap · W/X waypoint · click map = jump ")
        self.statusBar().addPermanentWidget(hint)

    def _fill_net_widgets(self):
        for c in (self.li_a, self.li_b, self.ba_center): c.addItems(self.names)
        gi = self.names.index("gen:ALL")
        self.li_a.setCurrentIndex(gi)
        self.li_b.setCurrentIndex(self.names.index("const:0.5"))
        self.ba_center.setCurrentIndex(gi)
        n = len(self.core.data["digits"][2])
        self.samp_spin.setRange(0, n - 1)

    # ---------------- wiring
    def _wire(self):
        self.mode_group.idClicked.connect(self.set_mode)
        self.metric_group.idClicked.connect(lambda _: (self.remap(), self.schedule_live()))
        self.focus_combo.currentTextChanged.connect(lambda _: self._on_focus())
        self.res_spin.valueChanged.connect(lambda _: self.remap())
        self.btn_remap.clicked.connect(self.remap)
        self.btn_dirs.clicked.connect(self._new_dirs)
        self.ba_center.currentTextChanged.connect(self._on_basin_center)
        for s in (self.pl_s, self.pl_t, self.vo_s, self.vo_t, self.vo_u,
                  self.li_m, self.ba_s, self.ba_t, *self.mix_sliders):
            s.valueChangedF.connect(self.schedule_live)
        self.li_a.currentTextChanged.connect(self.schedule_live)
        self.li_b.currentTextChanged.connect(self.schedule_live)
        self.samp_spin.valueChanged.connect(lambda _: self.schedule_live())
        self.canvas.mpl_connect("button_press_event", self.on_map_click)
        for i, key in enumerate("12345"):
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(lambda i=i: self.set_mode(i))
        for key, slot in (("Space", self._toggle_play), ("R", self.snap_nearest),
                          ("W", self.add_wp), ("X", self.del_wp)):
            sc = QShortcut(QKeySequence(key), self); sc.activated.connect(slot)

    # ---------------- mode / basis helpers
    def mode_name(self): return MODES[max(self.mode_group.checkedId(), 0)]
    def metric(self):    return METRICS[max(self.metric_group.checkedId(), 0)]
    def focus(self):     return self.focus_combo.currentText()

    def current_basis(self):
        if self.mode_name() == "BASIN" and self.basin is not None:
            return self.basin
        return dict(type="plane", key="plane", origin=self.mu,
                    d1=self.U1, d2=self.U2, extent=self.plane_extent)

    def line_endpoints(self):
        return (self.core.nets[self.li_a.currentText()],
                self.core.nets[self.li_b.currentText()])

    def current_vec(self):
        md = self.mode_name()
        if md == "PLANE":
            return self.mu + self.pl_s.value()*self.U1 + self.pl_t.value()*self.U2
        if md == "VOLUME":
            return (self.mu + self.vo_s.value()*self.U1 + self.vo_t.value()*self.U2
                    + self.vo_u.value()*self.U3)
        if md == "LINE":
            A, B = self.line_endpoints()
            return A + self.li_m.value() * (B - A)
        if md == "BASIN":
            b = self.basin
            return b["origin"] + self.ba_s.value()*b["d1"] + self.ba_t.value()*b["d2"]
        a = np.array([s.value() for s in self.mix_sliders])
        return (a[:, None] * self.M).sum(0) / a.sum() if a.sum() > 1e-9 else np.zeros_like(self.mu)

    def coords_of(self, vec):
        b = self.current_basis()
        if b["type"] == "plane":
            return float((vec - self.mu) @ self.U1), float((vec - self.mu) @ self.U2)
        return float((vec - b["origin"]) @ b["d1"]), float((vec - b["origin"]) @ b["d2"])

    def set_mode(self, idx):
        if not self.ready: return
        self.mode_btns[idx].setChecked(True)
        self.pages.setCurrentIndex(idx)
        if MODES[idx] == "BASIN":
            c = self.ba_center.currentText()
            if self.basin is None or self.basin["center"] != c:
                self._make_basin()
        self._sync_map_basis()
        self.schedule_live()

    def _make_basin(self):
        center = self.ba_center.currentText()
        A = self.core.nets[center]
        rng = np.random.RandomState(self.basin_seed)
        r1 = filter_normalize(rng.standard_normal(self.N_PARAMS), A, self.core.model)
        r2 = filter_normalize(rng.standard_normal(self.N_PARAMS), A, self.core.model)
        L = 0.5 * np.mean([np.linalg.norm(A - self.core.nets[n])
                           for n in self.trained if n != center])
        self.basin = dict(type="basin", key=f"basin:{center}:{self.basin_seed}",
                          origin=A, d1=r1, d2=r2, extent=(-L, L, -L, L), center=center)
        self.ba_s.set_range(-L, L); self.ba_s.set_value(0.0)
        self.ba_t.set_range(-L, L); self.ba_t.set_value(0.0)

    def _new_dirs(self):
        self.basin_seed += 1; self._make_basin()
        self._sync_map_basis(); self.schedule_live()

    def _on_basin_center(self):
        if self.ready and self.mode_name() == "BASIN":
            self._make_basin(); self._sync_map_basis(); self.schedule_live()

    def _on_focus(self):
        fd = "digits" if self.focus() == "ALL" else self.focus()
        self.samp_spin.setRange(0, max(len(self.core.data[fd][2]) - 1, 0))
        self._img_key = None
        self.remap(); self.schedule_live()

    # ---------------- presets
    def _preset_gen(self):
        for s in self.mix_sliders: s.set_value(0.0)
        self.mix_sliders[self.names.index("gen:ALL")].set_value(1.0)
        self.schedule_live()

    def _preset_uniform(self):
        k = len(self.trained)
        for s, name in zip(self.mix_sliders, self.names):
            s.set_value(1.0 / k if name in self.trained else 0.0)
        self.schedule_live()

    def _preset_half(self):
        for s in self.mix_sliders: s.set_value(0.5)
        self.schedule_live()

    # ---------------- map
    def _draw_static_map(self):
        cmap = mpl.colormaps["viridis"].copy(); cmap.set_bad("0.12")
        lo1, hi1, lo2, hi2 = self.plane_extent
        self.im = self.ax.imshow(np.full((21, 21), np.nan), origin="lower",
                                 extent=[lo1, hi1, lo2, hi2], aspect="auto",
                                 cmap=cmap, vmin=0.0, vmax=1.0, zorder=0)
        self.cbar = self.fig.colorbar(self.im, ax=self.ax, pad=0.02)
        self.cbar.set_label("map metric")
        self.scat = self.ax.scatter(self.P2[:, 0], self.P2[:, 1], c="red", s=25, zorder=3)
        self.anns = [self.ax.annotate(n, (x, y), fontsize=5.5, color="w",
                                      xytext=(3, 3), textcoords="offset points")
                     for n, (x, y) in zip(self.names, self.P2)]
        self.dot, = self.ax.plot([], [], "w+", ms=14, mew=2, zorder=5)
        self.path, = self.ax.plot([], [], "w--", lw=1, alpha=.7, zorder=2)
        self.trail_line, = self.ax.plot([], [], "-", color="orange", alpha=.45, lw=1, zorder=1)

    def _sync_map_basis(self):
        plane = self.current_basis()["type"] == "plane"
        self.scat.set_visible(plane)
        for a in self.anns: a.set_visible(plane)
        self.trail.clear(); self.trail_line.set_data([], [])
        self._render_waypoints()
        self.remap()

    def remap(self):
        if not self.ready: return
        basis = self.current_basis()
        met, ds, res = self.metric(), self.focus(), self.res_spin.value()
        key = (basis["key"], met, ds, res)
        self.ax.set_xlabel("PC1 (s)" if basis["type"] == "plane" else "random dir 1 (s)")
        self.ax.set_ylabel("PC2 (t)" if basis["type"] == "plane" else "random dir 2 (t)")
        self.ax.set_title(f"{basis['type']} slice · {met} · {ds}")
        self.cbar.set_label(met)
        if key in self.map_cache:
            H, extent, clim = self.map_cache[key]
            self.im.set_data(H); self.im.set_extent(list(extent)); self.im.set_clim(*clim)
            self.canvas.draw_idle()
            self.statusBar().showMessage("map: cached", 3000); return
        self.job += 1
        if self.map_worker and self.map_worker.isRunning():
            self.map_worker.stop = True
        self._H = np.full((res, res), np.nan); self._H_key = key
        lo1, hi1, lo2, hi2 = basis["extent"]
        self.im.set_data(self._H); self.im.set_extent([lo1, hi1, lo2, hi2])
        self.canvas.draw_idle()
        mi = METRICS.index(met); ds_list = self.core.ds_list
        def eval_fn(vec):
            m = self.core.metrics(vec)
            return float(np.mean([m[d][mi] for d in ds_list])) if ds == "ALL" else float(m[ds][mi])
        self._map_t0 = time.time()
        w = MapWorker(self.job, eval_fn, basis["origin"], basis["d1"], basis["d2"],
                      basis["extent"], res)
        w.row_ready.connect(self._on_map_row); w.done.connect(self._on_map_done)
        self.map_worker = w; w.start()
        self.statusBar().showMessage("computing map …")

    def _on_map_row(self, job, r, row, vmin, vmax):
        if job != self.job: return
        self._H[r] = row
        self.im.set_data(self._H); self.im.set_clim(vmin, vmax)
        self.canvas.draw_idle()

    def _on_map_done(self, job, completed):
        if job != self.job: return
        if completed:
            self.map_cache[self._H_key] = (self._H.copy(), self.im.get_extent(),
                                           self.im.get_clim())
            self.statusBar().showMessage(
                f"map done in {time.time() - self._map_t0:.1f}s (cached)", 4000)
        self.canvas.draw_idle()

    def on_map_click(self, ev):
        if not self.ready or ev.inaxes is not self.ax or ev.xdata is None: return
        x, y = float(ev.xdata), float(ev.ydata)
        basis = self.current_basis()
        lo1, hi1, lo2, hi2 = basis["extent"]
        span = max(hi1 - lo1, hi2 - lo2)
        for wp in self.waypoints:
            if wp["key"] == basis["key"] and np.hypot(wp["x"] - x, wp["y"] - y) < 0.03 * span:
                self._goto_wp(wp); return
        if basis["type"] == "plane" and self.mode_name() in ("PLANE", "VOLUME"):
            d = np.hypot(self.P2[:, 0] - x, self.P2[:, 1] - y)
            if d.min() < 0.03 * span:
                j = int(d.argmin())
                for s in (self.pl_s, self.vo_s): s.set_value(float(self.P2[j, 0]))
                for s in (self.pl_t, self.vo_t): s.set_value(float(self.P2[j, 1]))
                self.vo_u.set_value(float(self.P3[j]))
            else:
                for s in (self.pl_s, self.vo_s): s.set_value(np.clip(x, lo1, hi1))
                for s in (self.pl_t, self.vo_t): s.set_value(np.clip(y, lo2, hi2))
        elif basis["type"] == "basin":
            self.ba_s.set_value(np.clip(x, lo1, hi1))
            self.ba_t.set_value(np.clip(y, lo2, hi2))
        elif self.mode_name() == "LINE":
            A, B = self.line_endpoints()
            a2 = np.array([(A - self.mu) @ self.U1, (A - self.mu) @ self.U2])
            b2 = np.array([(B - self.mu) @ self.U1, (B - self.mu) @ self.U2])
            dv = b2 - a2
            t = float(np.clip(np.dot(np.array([x, y]) - a2, dv) / (dv @ dv + 1e-12), 0, 1))
            self.li_m.set_value(t)
        else:
            self.statusBar().showMessage("MIX mode: adjust the per-net sliders", 2000)
            return
        self.schedule_live()

    # ---------------- waypoints
    def add_wp(self):
        if not self.ready: return
        vec = self.current_vec(); x, y = self.coords_of(vec)
        self.waypoints.append(dict(
            key=self.current_basis()["key"], mode=self.mode_name(), vec=vec, x=x, y=y,
            n=len(self.waypoints) + 1,
            snap=dict(s=self.pl_s.value(), t=self.pl_t.value(), u=self.vo_u.value(),
                      vs=self.vo_s.value(), vt=self.vo_t.value(), m=self.li_m.value(),
                      bs=self.ba_s.value(), bt=self.ba_t.value(),
                      mix=[s.value() for s in self.mix_sliders])))
        self._render_waypoints(); self.canvas.draw_idle()

    def del_wp(self):
        if self.waypoints:
            self.waypoints.pop(); self._render_waypoints(); self.canvas.draw_idle()

    def _render_waypoints(self):
        for a in self.wp_artists:
            try: a.remove()
            except Exception: pass
        self.wp_artists = []
        if not self.ready: return
        key = self.current_basis()["key"]
        for wp in self.waypoints:
            if wp["key"] != key: continue
            ln, = self.ax.plot([wp["x"]], [wp["y"]], "o", ms=9, mfc="none",
                               mec="cyan", mew=1.5, zorder=4)
            tx = self.ax.annotate(f"W{wp['n']}", (wp["x"], wp["y"]), color="cyan",
                                  fontsize=6, zorder=4, xytext=(4, -2),
                                  textcoords="offset points")
            self.wp_artists += [ln, tx]

    def _goto_wp(self, wp):
        self.set_mode(MODES.index(wp["mode"]))
        s = wp["snap"]
        if wp["mode"] in ("PLANE", "VOLUME"):
            self.pl_s.set_value(s["s"]); self.pl_t.set_value(s["t"])
            self.vo_s.set_value(s["vs"]); self.vo_t.set_value(s["vt"]); self.vo_u.set_value(s["u"])
        elif wp["mode"] == "LINE":
            self.li_m.set_value(s["m"])
        elif wp["mode"] == "BASIN":
            self.ba_s.set_value(s["bs"]); self.ba_t.set_value(s["bt"])
        else:
            for sl, v in zip(self.mix_sliders, s["mix"]): sl.set_value(v)
        self.schedule_live()

    # ---------------- live update
    def schedule_live(self):
        self.live_timer.start()

    def _do_live(self):
        if not self.ready: return
        md = self.mode_name()
        vec = self.current_vec()
        m = self.core.metrics(vec)
        met, mi = self.metric(), METRICS.index(self.metric())
        ds_f = self.focus()
        vals = [m[d][mi] for d in self.core.ds_list]
        ymax = 1.0 if met != "loss" else max(1.1, max(vals) * 1.15)
        hi = self.core.ds_list.index(ds_f) if ds_f != "ALL" else -1
        self.bars.set_data(vals, ymax, f"{met} per dataset · avg {m['ALL'][mi]:.3f}", hi)

        d = np.linalg.norm(self.M - vec, axis=1); top = np.argsort(d)[:3]
        x, y = self.coords_of(vec)
        self.dot.set_data([x], [y])
        self.trail.append((x, y))
        tr = np.array(self.trail)
        if len(tr) > 1: self.trail_line.set_data(tr[:, 0], tr[:, 1])
        if md == "LINE":
            A, B = self.line_endpoints()
            pts = A + np.linspace(0, 1, 33)[:, None] * (B - A)
            self.path.set_data((pts - self.mu) @ self.U1, (pts - self.mu) @ self.U2)
        else:
            self.path.set_data([], [])

        if md == "BASIN":
            coord = f"s={self.ba_s.value():+.2f} t={self.ba_t.value():+.2f}"
        elif md == "LINE":
            coord = f"m={self.li_m.value():.2f}  A={self.li_a.currentText()} → B={self.li_b.currentText()}"
        elif md == "MIX":
            coord = "mix knobs (MIX page)"
        else:
            coord = f"s={self.pl_s.value():+.2f} t={self.pl_t.value():+.2f} u={self.vo_u.value():+.2f}"
        self.info.setPlainText("\n".join([
            f"mode {md:6s}  {coord}",
            f"nearest: {self.names[top[0]]} (d={d[top[0]]:.1f})",
            f"         {self.names[top[1]]} (d={d[top[1]]:.1f})",
            f"         {self.names[top[2]]} (d={d[top[2]]:.1f})",
            f"|w|={np.linalg.norm(vec):.1f}  acc={m['ALL'][0]:.1%} "
            f"conf={m['ALL'][1]:.2f} loss={m['ALL'][2]:.2f}"]))
        self._update_inspector(vec)
        self.canvas.draw_idle()

    def _update_inspector(self, vec):
        fd = "digits" if self.focus() == "ALL" else self.focus()
        i = self.samp_spin.value()
        x, ytrue = self.core.sample(fd, i)
        if (fd, i) != self._img_key:
            self._img_key = (fd, i)
            a = x.reshape(8, 8).astype(np.float32)
            a = (a - a.min()) / (float(np.ptp(a)) + 1e-9)
            buf = (a * 255).astype(np.uint8).tobytes()
            img = QImage(buf, 8, 8, 8, QImage.Format_Grayscale8)
            self.img_label.setPixmap(QPixmap.fromImage(img).scaled(
                self.img_label.size(), Qt.IgnoreAspectRatio, Qt.FastTransformation))
            self.img_label.setToolTip(f"{fd}[{i}]   true class = {ytrue}")
        self.probs.set_data(self.core.sample_probs(vec, fd, i), ytrue)

    # ---------------- play / snap / export
    def _toggle_play(self):
        self.playing = not self.playing
        self.btn_play.setText("Pause" if self.playing else "Play")
        self.play_timer.start() if self.playing else self.play_timer.stop()

    def _step(self):
        sl = {"PLANE": self.pl_t, "VOLUME": self.vo_t,
              "LINE": self.li_m, "BASIN": self.ba_t}.get(self.mode_name())
        if sl is None: return
        step = (sl.hi - sl.lo) / 120 * self.play_dir
        v = sl.value() + step
        if v >= sl.hi or v <= sl.lo:
            self.play_dir *= -1; v = float(np.clip(v, sl.lo, sl.hi))
        sl.set_value(v); self.schedule_live()

    def snap_nearest(self):
        if not self.ready: return
        d = np.linalg.norm(self.M - self.current_vec(), axis=1); j = int(d.argmin())
        md = self.mode_name()
        if md in ("PLANE", "VOLUME"):
            for s in (self.pl_s, self.vo_s): s.set_value(float(self.P2[j, 0]))
            for s in (self.pl_t, self.vo_t): s.set_value(float(self.P2[j, 1]))
            self.vo_u.set_value(float(self.P3[j]))
        elif md == "LINE":
            self.li_a.setCurrentText(self.names[j]); self.li_m.set_value(0.0)
        elif md == "BASIN":
            self.ba_center.setCurrentText(self.names[j])
        else:
            for s in self.mix_sliders: s.set_value(0.0)
            self.mix_sliders[j].set_value(1.0)
        self.schedule_live()

    def save_png(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save PNG", "weightspace.png", "PNG (*.png)")
        if fn: self.fig.savefig(fn, dpi=160)

    def save_vec(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save weight vector", "point.npz",
                                            "NumPy (*.npz)")
        if fn:
            np.savez(fn, vec=self.current_vec(), mode=self.mode_name())
            self.statusBar().showMessage(f"saved {fn}", 4000)

    def closeEvent(self, ev):
        self.play_timer.stop()
        if self.map_worker and self.map_worker.isRunning():
            self.map_worker.stop = True
            self.map_worker.wait(2000)
        super().closeEvent(ev)

# ================================================================ entry
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    print(f"device: {DEVICE}")
    core = Core()
    ensure_nets(core)          # trains on first launch (modal progress), else loads cache
    win = MainWindow(core)
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()