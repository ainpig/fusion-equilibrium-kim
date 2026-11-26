"""v17 MAST: frames solved as inboard-limited take betaN (and the q95 shape factor) from the run's own
diverted frames.

v14 found that a frame whose separatrix rests on the inner support (q_physical_v14.limited_flags) is an
artefact of the seed solve -- every frame of the challenge is diverted -- and that its q95 shape factor is
wrong; v16 gave those frames the shape factor interpolated from the shot's diverted frames. Most of them
(84%) precede the shot's first diverted frame, so that "interpolation" is a constant carried back from
the first diverted frame into the ramp-up.

v17 asks instead what the run itself gives a frame of the same input state when it does resolve the
geometry. A gradient-boosted tree model (CatBoost) is fitted on the run's DIVERTED frames only, from
geometry-independent inputs -- coil, solenoid, TF and plasma currents and their rates, Thomson Te / ne /
pressure statistics and the time in the shot -- shot-grouped five-fold, so a shot is predicted by trees
that never saw it. A limited frame takes model x ratio(t), where ratio is the shot's own (ours / model)
over its diverted frames, interpolated in time (clipped to 0.5-2; 1 if the shot has none), then a
five-frame running median. Diverted frames and the flux are left as they are.

On the run's own diverted ramp-up frames, hidden and refilled (fresh 1206-shot run, no label read), this
reproduces what the run gives them with R2 0.95 for betaN and 0.95 for the shape factor; the shot-local
interpolation reaches 0.25 and 0.74. On the limited frames betaN moves by 0.36 rms, q95 by 1.0 rms. The
SHAP values of the betaN model (CatBoost TreeSHAP) put the Thomson Te maximum, the solenoid current and
its rate, the pressure / Ip^2 proxy and the P4 / P5 vertical-field coils first -- the inputs a Shafranov
force balance ties betaN to.

Nothing here reads a MAST target, a score or another run: the trees are fitted on this run's own
predictions, every run anew.
"""
import numpy as np

E_CHARGE = 1.602176634e-19
COILS = ["p2l", "p2u", "p3l", "p3u", "p4l", "p4u", "p5l", "p5u", "p6l", "p6u", "efps", "sol", "tf"]
FEATURES = ["ip", "dip", "ip_frac", "t_frac", "t", "p2l", "p2u", "p3l", "p3u", "p4l", "p4u", "p5l", "p5u",
            "p6l", "p6u", "efps", "sol", "dsol", "tf", "te_max", "ne_mean", "pe_q95", "pe_int", "te_width",
            "te_rpeak", "ip_ma", "bt", "pe_over_ip2"]
FOLDS = 5
RATIO_CLIP = (0.5, 2.0)
FLOOR = {"betaN": 0.05, "y": 0.5}       # model values below this do not set the shot's ratio
MEDIAN_WINDOW = 5
TREES = dict(iterations=1500, depth=8, learning_rate=0.05, loss_function="RMSE", thread_count=12)
SEEDS = {"betaN": 0, "y": 100}
THOMSON_GAP_MS = 10.0


def _array(value):
    """A float array from a parquet cell (a float array, or an object array / list of per-pulse arrays)."""
    if isinstance(value, np.ndarray) and value.dtype != object:
        return value.astype(float)
    items = list(value)
    if items and np.ndim(items[0]) > 0:
        return np.array([np.asarray(v, float) for v in items], dtype=float)
    return np.asarray(items, float)


def _interp(t, x, tq):
    t = _array(t)
    x = _array(x)
    ok = np.isfinite(t) & np.isfinite(x)
    if ok.sum() < 2:
        return np.full(len(tq), np.nan)
    o = np.argsort(t[ok])
    return np.interp(tq, t[ok][o], x[ok][o])


def frame_features(row):
    """(T, len(FEATURES)) geometry-independent inputs of one shot at its EFIT times (input columns only)."""
    t = _array(row["efit_times"])
    tm = _array(row["magnetics_time"])
    ip = _interp(tm, row["magnetics_plasma_current"], t)
    f = dict(ip=ip, dip=np.gradient(ip, t) if len(t) > 1 else np.zeros(len(t)),
             ip_frac=ip / np.nanmax(np.abs(ip)), t_frac=(t - t.min()) / max(np.ptp(t), 1e-9), t=t)
    for c in COILS:
        f[c] = _interp(tm, row[f"magnetics_{c}_current"], t)
    f["dsol"] = np.gradient(f["sol"], t) if len(t) > 1 else np.zeros(len(t))
    # the core Thomson pulse nearest each frame, as shipped (finite, positive channels only)
    tt = _array(row["thomson_core_times"])
    te = _array(row["thomson_core_Te"])
    ne = _array(row["thomson_core_ne"])
    tr = _array(row["thomson_core_R"])
    if tr.ndim == 2:
        tr = np.nanmedian(tr, axis=0)
    kin = {k: np.full(len(t), np.nan) for k in ("te_max", "ne_mean", "pe_q95", "pe_int", "te_width", "te_rpeak")}
    for k in range(len(t)):
        j = int(np.nanargmin(np.abs(tt - t[k]))) if len(tt) else -1
        if j < 0 or abs(tt[j] - t[k]) > THOMSON_GAP_MS:
            continue
        a, b = te[j], ne[j]
        ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
        if ok.sum() < 10:
            continue
        p = 2 * E_CHARGE * a[ok] * b[ok]
        rr = tr[ok] if len(tr) == len(a) else np.linspace(0.25, 1.45, ok.sum())
        kin["te_max"][k] = np.quantile(a[ok], .97)
        kin["ne_mean"][k] = np.mean(b[ok])
        kin["pe_q95"][k] = np.quantile(p, .95)
        o = np.argsort(rr)
        kin["pe_int"][k] = np.trapezoid(p[o], rr[o])
        hot = rr[a[ok] > 0.3 * np.quantile(a[ok], .97)]
        kin["te_width"][k] = np.ptp(hot) if len(hot) > 1 else np.nan
        kin["te_rpeak"][k] = rr[int(np.argmax(a[ok]))]
    f.update(kin)
    f["ip_ma"] = np.abs(ip) / 1000.0
    f["bt"] = (np.abs(f["tf"]) * 1000 * 24 * 2e-7) / 0.85
    f["pe_over_ip2"] = f["pe_int"] / np.maximum(f["ip_ma"], 0.05) ** 2
    return np.column_stack([np.asarray(f[n], float) for n in FEATURES])


def shot_folds(n_shots):
    """Shot -> fold (0..FOLDS-1), fixed by the shot order alone."""
    order = np.arange(n_shots)
    np.random.default_rng(0).shuffle(order)
    folds = np.empty(n_shots, int)
    folds[order] = np.arange(n_shots) % FOLDS
    return folds


def out_of_fold(X, y, train, fold, seed):
    """Each frame predicted by trees fitted on the training frames of the other folds' shots."""
    from catboost import CatBoostRegressor
    out = np.full(len(y), np.nan)
    for k in range(FOLDS):
        model = CatBoostRegressor(**TREES, random_seed=seed + k, verbose=0, allow_writing_files=False)
        tr = train & (fold != k)
        model.fit(X[tr], y[tr])
        te = fold == k
        out[te] = model.predict(X[te])
    return out


def _running_median(x, window=MEDIAN_WINDOW):
    if len(x) < 3:
        return x.copy()
    h = window // 2
    p = np.concatenate([np.full(h, x[0]), x, np.full(h, x[-1])])
    return np.array([np.median(p[i:i + window]) for i in range(len(x))])


def fill(target, model, limited, floor):
    """The shot's limited frames := model x ratio(t) (ratio over its diverted frames), smoothed."""
    target = np.asarray(target, float)
    limited = np.asarray(limited, bool)
    if not limited.any():
        return target.copy()
    idx = np.arange(len(target))
    ok = ~limited & (model > floor) & np.isfinite(target)
    ratio = np.interp(idx, idx[ok], target[ok] / model[ok]) if ok.any() else np.ones(len(target))
    series = np.where(limited, model * np.clip(ratio, *RATIO_CLIP), target)
    return np.where(limited, _running_median(series), target)
