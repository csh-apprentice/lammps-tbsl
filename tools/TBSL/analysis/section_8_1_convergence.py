import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


from ._config import OUTPUT_ROOT

FIG_DIR = OUTPUT_ROOT / "figs" / "convergence"

CUTOFF_COLORS = {
    5: "#4C566A",
    10: "#B04A5A",
    20: "#7A6C9D",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def save_current_figure(filename: str) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), bbox_inches="tight")
    plt.close()


def convergence_rate(plist: np.ndarray) -> np.ndarray:
    rates = []
    for i in range(1, len(plist) - 1):
        enew = abs(plist[i] - plist[-1])
        eold = abs(plist[i - 1] - plist[-1])
        rates.append(np.log2(eold / enew))
    return np.asarray(rates, dtype=float)


def plot_cutoff_convergence() -> None:
    # These reference sequences are taken directly from convergence.ipynb.
    plist_5 = np.array(
        [
            20.21857502469134360,
            13.21026604801576809,
            13.18744320051363417,
            13.17992368090338573,
            13.18033854381826586,
            13.17980779207649178,
            13.18002042877433233,
            13.17989211410229089,
            13.17988440991470789,
            13.17990315510193788,
        ]
    )
    plist_10 = np.array(
        [
            20.66621185906510405,
            13.22491638248749091,
            13.19584303047391316,
            13.19103721292000664,
            13.18985023840785331,
            13.18955175407649172,
            13.18948048879579460,
            13.18946007863640624,
            13.18945607669565412,
            13.18945394067020516,
        ]
    )
    plist_20 = np.array(
        [
            20.68238009878830397,
            13.22560641322502484,
            13.19635954864375904,
            13.19154423494749651,
            13.19036096608291153,
            13.19006643393196399,
            13.18999285631440621,
            13.18997445893531584,
            13.18996986768345003,
            13.18996833648377098,
        ]
    )

    rates_5 = convergence_rate(plist_5)
    rates_10 = convergence_rate(plist_10)
    rates_20 = convergence_rate(plist_20)
    timestep_exponents = np.arange(4, -4, -1)

    plt.figure(figsize=(4.4, 3.4))
    plt.plot(timestep_exponents, rates_5, color=CUTOFF_COLORS[5], marker="o", linewidth=2.0, markersize=4, label="Cutoff 5 Å")
    plt.plot(timestep_exponents, rates_10, color=CUTOFF_COLORS[10], marker="^", linewidth=2.0, markersize=4, label="Cutoff 10 Å")
    plt.plot(timestep_exponents, rates_20, color=CUTOFF_COLORS[20], marker="s", linewidth=2.0, markersize=4, label="Cutoff 20 Å")
    plt.axhline(y=2.0, color="black", linestyle="--", linewidth=1.5)
    plt.xlabel(r"Timestep Exponent $n$ in $\Delta t = 2^n$ fs")
    plt.ylabel("Convergence Rate")
    plt.xticks(timestep_exponents)
    plt.legend(frameon=False)
    save_current_figure("Cutoff.pdf")

    print("Cutoff convergence summary")
    print(f"5 A cutoff:  {rates_5}")
    print(f"10 A cutoff: {rates_10}")
    print(f"20 A cutoff: {rates_20}")


def plot_lj_cut_potential() -> None:
    epsilon = 8.314462618 * 116.81 * 0.00023901
    sigma = 3.401
    rmin = 0.1
    rmax = 25.0
    rcut = 10.0

    r = np.linspace(rmin, rmax, 2000)
    potential = 4.0 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
    truncated = np.where(r <= rcut, potential, 0.0)

    plt.figure(figsize=(4.4, 3.4))
    plt.plot(r, truncated, color="black", linewidth=2.0)
    plt.axvline(5.0, color=CUTOFF_COLORS[5], linestyle="--", linewidth=1.5, label="Cutoff 5 Å")
    plt.axvline(10.0, color=CUTOFF_COLORS[10], linestyle="--", linewidth=1.5, label="Cutoff 10 Å")
    plt.axvline(20.0, color=CUTOFF_COLORS[20], linestyle="--", linewidth=1.5, label="Cutoff 20 Å")
    plt.xlim(0.0, rmax)
    plt.ylim(-epsilon, epsilon)
    plt.xlabel("r (Å)")
    plt.ylabel(r"$U(r)$ (kcal mol$^{-1}$)")
    plt.legend(frameon=False)
    save_current_figure("Potential.pdf")

    print("Potential summary")
    print(f"epsilon = {epsilon:.9f} kcal/mol")
    print(f"sigma = {sigma:.3f} A, truncation shown at r_c = {rcut:.1f} A")


def main() -> None:
    plot_cutoff_convergence()
    plot_lj_cut_potential()


if __name__ == "__main__":
    main()
