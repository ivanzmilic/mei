#!/usr/bin/env python
"""fits_inversion_videos.py -- render videos from a Milne-Eddington inversion cube.

Sibling of npz_inversion_videos.py, but for the FITS output written by
invert_mihi_6300.py / invert_mihi_5892.py.  That file packs TWO HDUs:

    HDU 0  model  (Nt, Ny, Nx, 9)      the 9 Milne-Eddington parameters
    HDU 1  synth  (Nt, Ny, Nx, 4, L)   synthetic Stokes I,Q,U,V spectra  (optional)

The 9 ME columns (see the model_guess in invert_mihi_*.py) are
    0 |B| [G]      1 inclination [rad]   2 azimuth [rad]      3 vlos [km/s]
    4 Doppler width [nm]                 5 eta0 (line opacity)
    6 damping      7 S0                  8 S1
and the continuum intensity is S0 + S1.

Unlike the .npz sibling the FITS holds NO observed spectra, so there is no
obs/fit/residual panel -- the "spectra" clip shows the *synthetic* Stokes maps.

Two videos are written:
    <prefix>_params.mp4   physical parameter maps (Blos, Btrans, azimuth,
                          vlos, Doppler width, continuum intensity)
    <prefix>_stokes.mp4   synthetic Stokes I,Q,U,V maps at a few wavelengths

KEY POINT (same as the npz sibling): every panel's colour limits are computed
ONCE over ALL time and held fixed for the whole clip, so colours/colourbars
never jump between frames.  Signed quantities (Blos, vlos, Q/U/V) use a
symmetric scale; azimuth is fixed to [0, pi].

The grid and wavelength count are read straight from the file, so the script
works for any cube written in this 2-HDU layout, not just the MiHI default.

Usage:
    python fits_inversion_videos.py                                   # defaults
    python fits_inversion_videos.py --fits /path/results.fits --fps 8
    python fits_inversion_videos.py --params-only
    python fits_inversion_videos.py --stokes-only --dwl 30 --prefix run3
"""
import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FITS = "/dat/milic/inverted_results_mihi_me.fits"


# ----------------------------------------------------------------------- helpers
def sym(lo, hi):
    """Symmetric limits about zero spanning the larger magnitude of lo/hi."""
    m = max(abs(lo), abs(hi))
    return -m, m


def limits(a, plo, phi, symmetric=False):
    """Fixed colour limits from percentiles over the WHOLE cube (all t, y, x)."""
    lo = float(np.nanpercentile(a, plo))
    hi = float(np.nanpercentile(a, phi))
    return sym(lo, hi) if symmetric else (lo, hi)


def save_animation(fig, update, n_frames, out_path, fps):
    """Save a FuncAnimation to out_path, picking the writer from the extension.

    .mp4/.avi -> ffmpeg (falls back to an animated .gif if ffmpeg is missing)."""
    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    ext = os.path.splitext(out_path)[1].lower()
    try:
        if ext == ".gif":
            writer = PillowWriter(fps=fps)
        else:
            writer = FFMpegWriter(fps=fps, codec="libx264",
                                  extra_args=["-pix_fmt", "yuv420p"])
        anim.save(out_path, writer=writer, dpi=fig.get_dpi())
    except (FileNotFoundError, RuntimeError) as e:
        gif = os.path.splitext(out_path)[0] + ".gif"
        print("  ffmpeg unavailable (%s); writing %s instead" % (e, gif))
        anim.save(gif, writer=PillowWriter(fps=fps), dpi=fig.get_dpi())
        out_path = gif
    plt.close(fig)
    print("wrote %s  (%d frames, %.1f fps)" % (out_path, n_frames, fps))


def add_panel(ax, frame0, cmap, vmin, vmax, title):
    """Create one imshow panel (transposed + origin lower, as in the map plots)."""
    im = ax.imshow(frame0.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    return im


# ----------------------------------------------------------------------- params
def make_params_video(model, out_path, fps, pct, dpi, cadence):
    Nt = model.shape[0]
    plo, phi = pct, 100.0 - pct

    B, inc, azi, vlos = (model[..., i] for i in range(4))
    dopp = model[..., 4]
    cont = model[..., 7] + model[..., 8]
    Blos = B * np.cos(inc)
    Btr = B * np.sin(inc)

    # (name, cube, cmap, (vmin, vmax)) -- limits fixed over all time
    panels = [
        ("Blos [G]",            Blos, "PuOr",     limits(Blos, plo, phi, symmetric=True)),
        ("Btrans [G]",          Btr,  "cividis",  (0.0, float(np.nanpercentile(Btr, phi)))),
        ("azimuth [rad]",       azi,  "twilight", (0.0, np.pi)),
        ("vlos [km/s]",         vlos, "bwr",      limits(vlos, plo, phi, symmetric=True)),
        ("Doppler width [nm]",  dopp, "inferno",  limits(dopp, plo, phi)),
        ("continuum I",         cont, "inferno",  limits(cont, plo, phi)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), dpi=dpi, constrained_layout=True)
    axes = axes.ravel()
    ims = [add_panel(axes[k], p[1][0], p[2], p[3][0], p[3][1], p[0])
           for k, p in enumerate(panels)]
    sup = fig.suptitle("", fontsize=13)

    def update(t):
        for im, (_, cube, _, _) in zip(ims, panels):
            im.set_data(cube[t].T)
        sup.set_text("ME parameters -- frame %d / %d   t = %.1f s (fixed scales)"
                     % (t, Nt - 1, t * cadence))
        return ims

    print("PARAMS video (%d frames):" % Nt)
    save_animation(fig, update, Nt, out_path, fps)


# ----------------------------------------------------------------------- stokes
def make_stokes_video(synth, out_path, fps, pct, dpi, dwl, cadence):
    Nt, _, _, _, L = synth.shape
    plo, phi = pct, 100.0 - pct

    core = L // 2
    wl_idx = sorted(set([max(core - dwl, 0), core, min(core + dwl, L - 1)]))
    names = ["I", "Q", "U", "V"]
    cmaps = ["magma", "seismic", "seismic", "seismic"]

    # fixed limits per (Stokes, wavelength); Q/U/V symmetric about zero
    lim = {}
    for s in range(4):
        for w in wl_idx:
            lim[(s, w)] = limits(synth[:, :, :, s, w], plo, phi, symmetric=(s > 0))

    nrow, ncol = 4, len(wl_idx)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.0 * nrow),
                             dpi=dpi, constrained_layout=True)
    axes = np.atleast_2d(axes).reshape(nrow, ncol)
    ims = {}
    for r in range(nrow):
        for c, w in enumerate(wl_idx):
            vmin, vmax = lim[(r, w)]
            title = "%s @ core%+d" % (names[r], w - core) if w != core else "%s @ core" % names[r]
            ims[(r, c)] = add_panel(axes[r, c], synth[0, :, :, r, w], cmaps[r],
                                    vmin, vmax, title)
    sup = fig.suptitle("", fontsize=13)

    def update(t):
        arts = []
        for r in range(nrow):
            for c, w in enumerate(wl_idx):
                ims[(r, c)].set_data(synth[t, :, :, r, w].T)
                arts.append(ims[(r, c)])
        sup.set_text("synthetic Stokes -- frame %d / %d   t = %.1f s (fixed scales)"
                     % (t, Nt - 1, t * cadence))
        return arts

    print("STOKES video (%d frames, wavelengths %s):" % (Nt, wl_idx))
    save_animation(fig, update, Nt, out_path, fps)


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fits", default=DEFAULT_FITS,
                    help="inversion result FITS (HDU0 model, HDU1 synthetic Stokes)")
    ap.add_argument("--outdir", default=HERE, help="where to write the videos")
    ap.add_argument("--prefix", default="inversion_mihi", help="output filename prefix")
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--pct", type=float, default=1.0,
                    help="colour-limit percentile (uses pct .. 100-pct)")
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--dwl", type=int, default=40,
                    help="wavelength offset from core for the Stokes columns")
    ap.add_argument("--cadence", type=float, default=3.0,
                    help="time per snapshot [s], shown as an elapsed-time counter")
    ap.add_argument("--params-only", action="store_true")
    ap.add_argument("--stokes-only", action="store_true")
    args = ap.parse_args()

    with fits.open(args.fits) as hdul:
        model = np.asarray(hdul[0].data, dtype=np.float64)
        synth = np.asarray(hdul[1].data, dtype=np.float64) if len(hdul) > 1 and hdul[1].data is not None else None

    if model.ndim != 4 or model.shape[-1] < 9:
        raise SystemExit("HDU0 model has shape %s; expected (Nt, Ny, Nx, >=9)" % (model.shape,))
    Nt, Ny, Nx = model.shape[:3]
    print("loaded %s | model %s%s"
          % (args.fits, model.shape,
             "" if synth is None else " | synth %s" % (synth.shape,)))

    os.makedirs(args.outdir, exist_ok=True)

    if not args.stokes_only:
        out = os.path.join(args.outdir, "%s_params.mp4" % args.prefix)
        make_params_video(model, out, args.fps, args.pct, args.dpi, args.cadence)

    if not args.params_only:
        if synth is None:
            print("no synthetic-Stokes HDU in the file -- skipping the Stokes video")
        else:
            out = os.path.join(args.outdir, "%s_stokes.mp4" % args.prefix)
            make_stokes_video(synth, out, args.fps, args.pct, args.dpi, args.dwl, args.cadence)


if __name__ == "__main__":
    main()
