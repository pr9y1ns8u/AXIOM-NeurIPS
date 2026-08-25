"""
analyze.py -- everything downstream of the GPU run. Reads curves.npz only.

Answers three questions probe.py does not:
  1. Is the displacement factor layer-invariant, or does it vary?
  2. Does attention mass fail RANDOMLY, or does it systematically
     underestimate damage in deep layers?
  3. What is the correlation in the budget range the literature actually
     reports (B = 64..1024), rather than averaged over degenerate extremes?

    python analyze.py --curves curves.npz
"""

import argparse
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--curves", default="figs/curves.npz")
    p.add_argument("--lo", type=int, default=64, help="low end of the realistic budget band")
    p.add_argument("--hi", type=int, default=1024, help="high end of the realistic budget band")
    args = p.parse_args()

    z = np.load(args.curves, allow_pickle=True)
    grid, A, O, G = z["grid"], z["D_att"], z["D_out"], z["D_disp"]
    L = A.shape[0]
    depth = np.arange(L)

    print(f"model: {z['model']}   layers: {L}   budgets: {len(grid)}")
    print(f"headline rho (all budgets): {float(z['rho']):.3f}")

    # ---------------------------------------------------------------
    # 1. the realistic band -- what the literature actually reports at
    # ---------------------------------------------------------------
    band = (grid >= args.lo) & (grid <= args.hi)
    rho_band = np.array([spearmanr(A[:, b], O[:, b]).statistic
                         for b in np.where(band)[0]])
    rho_band = rho_band[~np.isnan(rho_band)]
    print(f"\nrho restricted to B in [{args.lo}, {args.hi}]  ({band.sum()} budgets):")
    print(f"  mean {rho_band.mean():.3f}   min {rho_band.min():.3f}   max {rho_band.max():.3f}")
    print("  ^ this is the number a reviewer will care about, not the all-budget mean")

    # ---------------------------------------------------------------
    # 2. is the displacement factor layer-invariant?
    # ---------------------------------------------------------------
    cv = np.nanstd(G, axis=0) / (np.nanmean(G, axis=0) + 1e-12)
    print(f"\ndisplacement CV across layers: mean {np.nanmean(cv):.3f}, "
          f"at B={grid[band][0]}: {cv[band][0]:.3f}, at B={grid[band][-1]}: {cv[band][-1]:.3f}")
    print("  small -> attention mass alone ranks layers fine")
    print("  large -> the geometric factor carries real, ignored information")

    # ---------------------------------------------------------------
    # 3. THE KEY TEST: is the failure systematic with depth?
    # ---------------------------------------------------------------
    # If D_disp rises with layer index, attention mass does not merely fail
    # noisily -- it UNDERESTIMATES damage in deep layers by a predictable
    # amount. That is a directional bias and a far sharper claim.
    depth_rho = np.array([
        spearmanr(depth[np.isfinite(G[:, b])], G[np.isfinite(G[:, b]), b]).statistic
        if np.isfinite(G[:, b]).sum() > 2 else np.nan
        for b in range(len(grid))])
    dr_band = depth_rho[band]
    dr_band = dr_band[~np.isnan(dr_band)]
    print(f"\nSpearman(layer depth, displacement) over B in [{args.lo}, {args.hi}]:")
    print(f"  mean {dr_band.mean():.3f}   min {dr_band.min():.3f}   max {dr_band.max():.3f}")
    if abs(dr_band.mean()) > 0.5:
        direction = "deeper layers suffer MORE" if dr_band.mean() > 0 else "deeper layers suffer LESS"
        print(f"  => SYSTEMATIC: {direction} output damage per unit attention mass dropped.")
        print("     This is a directional bias, not noise. Lead the paper with it.")
    else:
        print("  => no strong monotone depth trend; the failure looks unstructured.")

    # ---------------------------------------------------------------
    # figure: displacement vs depth at a few budgets
    # ---------------------------------------------------------------
    picks = [i for i in np.where(band)[0][:: max(band.sum() // 4, 1)]][:4]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))

    for i in picks:
        ax[0].plot(depth, G[:, i], marker="o", ms=3, lw=1, label=f"B={grid[i]}")
    ax[0].set_xlabel("layer depth")
    ax[0].set_ylabel(r"displacement  $\|\mu - \tilde o\|$")
    ax[0].set_title("the factor attention mass ignores", fontsize=9)
    ax[0].legend(fontsize=7)

    ax[1].plot(grid, depth_rho, marker="o", ms=3, lw=1, color="C3")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].axvspan(args.lo, args.hi, alpha=0.12, color="C0", label="reported budget range")
    ax[1].set_xscale("log"); ax[1].set_ylim(-1, 1)
    ax[1].set_xlabel("budget $B$")
    ax[1].set_ylabel(r"Spearman(depth, displacement)")
    ax[1].set_title("is the bias systematic with depth?", fontsize=9)
    ax[1].legend(fontsize=7)

    fig.savefig("figs/displacement.pdf", bbox_inches="tight")
    print("\nwrote figs/displacement.pdf")


if __name__ == "__main__":
    main()