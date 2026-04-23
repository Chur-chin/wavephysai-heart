"""
generate_all_figures.py
Reproduce all paper figures (Fig. 1-5).

Usage:
    python figures/generate_all_figures.py          # all figures
    python figures/generate_all_figures.py --fig 4  # single figure
"""

import sys, os, argparse, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model       import HeartEMSystem
from src.lyapunov    import compute_lyapunov
from src.bifurcation import bifurcation_diagram, poincare_section

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.family": "serif", "font.size": 10,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})


# ── Fig 1: System schematic ──────────────────────────────────────────
def fig1():
    print("Fig 1...")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")

    def box(x, y, w, h, label, color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.12",
                                    facecolor=color,edgecolor="k",lw=1.8))
        ax.text(x+w/2,y+h/2,label,ha="center",va="center",
                fontsize=9,fontweight="bold",multialignment="center")

    box(0.5,5.0,2.8,1.8,"EM Field\n"+r"$\omega_E=2.0$ rad/s","#AED6F1")
    box(3.6,2.3,2.8,1.8,"Heart Oscillator\n"+r"$\omega_0\in[0.3,2.5]$ rad/s","#A9DFBF")
    box(6.7,5.0,2.8,1.8,"Vascular Wave\n"+r"$\omega_p=1.5$ rad/s","#F9E79F")

    ax.annotate("",xy=(3.6,3.1),xytext=(1.9,5.0),
                arrowprops=dict(arrowstyle="<->",color="#2471A3",lw=2))
    ax.annotate("",xy=(6.7,3.1),xytext=(6.5,5.0),
                arrowprops=dict(arrowstyle="<->",color="#B7950B",lw=2))
    ax.text(2.3,4.2,r"$g_1$",fontsize=14,color="#2471A3",fontweight="bold")
    ax.text(7.0,4.2,r"$g_2$",fontsize=14,color="#B7950B",fontweight="bold")

    for i,eq in enumerate([r"$\ddot{E}=-\omega_E^2E-\alpha E^3+g_1x$",
                            r"$\ddot{x}=-\omega_0^2x-\mu(x^2-1)\dot{x}+g_1E+g_2p$",
                            r"$\ddot{p}=-\omega_p^2p+g_2x$"]):
        ax.text(5.0,1.5-i*0.58,eq,ha="center",fontsize=8.5)

    ax.set_title("FIG. 1.  Multi-scale coupled oscillator system",fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT,"fig1_system.png"))
    plt.close()
    print("  -> fig1_system.png")


# ── Fig 2: Lyapunov + phase portrait ────────────────────────────────
def fig2():
    print("Fig 2 (slow ~2 min)...")
    g_vals = np.linspace(0.05,1.5,30)
    lams   = [compute_lyapunov(HeartEMSystem(w0=1.0,g1=g),T=200) for g in g_vals]
    lams   = np.array(lams)

    sol_c  = HeartEMSystem(w0=1.0,g1=0.8).integrate(T=350)
    skip   = sol_c.y.shape[1]//2

    fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(13,4.5))
    ax1.axhline(0,color="gray",lw=0.8,ls="--",alpha=0.7)
    ax1.fill_between(g_vals,lams,0,where=lams>0,alpha=0.3,color="tomato",label=r"Chaotic")
    ax1.fill_between(g_vals,lams,0,where=lams<=0,alpha=0.3,color="royalblue",label=r"Regular")
    ax1.plot(g_vals,lams,"k-",lw=1.5)
    ax1.set_xlabel(r"$g_1$"); ax1.set_ylabel(r"$\lambda_{\max}$")
    ax1.set_title(r"(a) Lyapunov exponent"); ax1.legend(fontsize=8)
    ax2.plot(sol_c.y[2,skip:],sol_c.y[3,skip:],"b-",lw=0.2,alpha=0.7)
    ax2.set_xlabel(r"$x$"); ax2.set_ylabel(r"$\dot{x}$")
    ax2.set_title(r"(b) Strange attractor $(x,\dot{x})$")
    ax3.plot(sol_c.y[0,skip:],sol_c.y[1,skip:],"r-",lw=0.2,alpha=0.7)
    ax3.set_xlabel(r"$E$"); ax3.set_ylabel(r"$\dot{E}$")
    ax3.set_title(r"(c) Strange attractor $(E,\dot{E})$")
    plt.suptitle(r"FIG. 2.  Lyapunov spectrum and strange attractors ($\omega_0=1.0$)",fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT,"fig2_lyapunov.png"))
    plt.close()
    print("  -> fig2_lyapunov.png")


# ── Fig 3: Bifurcation diagram ───────────────────────────────────────
def fig3():
    print("Fig 3...")
    g_out,x_peaks=bifurcation_diagram(w0=1.0,g1_values=np.linspace(0.05,1.5,100))
    fig,ax=plt.subplots(figsize=(8,5))
    ax.scatter(g_out,x_peaks,s=0.5,c="navy",alpha=0.5)
    ax.set_xlabel(r"$g_1$",fontsize=11); ax.set_ylabel(r"$x_{\rm peak}$",fontsize=11)
    ax.set_title(r"FIG. 3.  Bifurcation diagram ($\omega_0=1.0$)",fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT,"fig3_bifurcation.png"))
    plt.close()
    print("  -> fig3_bifurcation.png")


# ── Fig 4: g_crit vs w0 ─────────────────────────────────────────────
def fig4():
    print("Fig 4 (slow ~3 min)...")
    from src.scaling_law import compute_scaling_law
    w0_arr,gcrit_arr,coeffs=compute_scaling_law(verbose=True,T=200)
    w0f=np.linspace(w0_arr.min(),w0_arr.max(),200)
    a,b,c=coeffs
    lbl=fr"Fit: $g_{{\rm crit}}={a:.3f}\omega_0^2+{b:.3f}\omega_0+{c:.3f}$"
    fig,ax=plt.subplots(figsize=(7,5))
    ax.plot(w0_arr,gcrit_arr,"ko",ms=8,zorder=5,label="Numerical data")
    ax.plot(w0f,np.polyval(coeffs,w0f),"r-",lw=2,label=lbl)
    ax.set_xlabel(r"$\omega_0$ (rad/s)",fontsize=11)
    ax.set_ylabel(r"$g_{\rm crit}$",fontsize=11)
    ax.set_title(r"FIG. 4.  Scaling law $g_{\rm crit}(\omega_0)$",fontsize=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT,"fig4_gcrit.png"))
    plt.close()
    print("  -> fig4_gcrit.png")


# ── Fig 5: Time series + Poincare ───────────────────────────────────
def fig5():
    print("Fig 5...")
    fig,axes=plt.subplots(2,2,figsize=(11,6))
    for col,(gv,label) in enumerate([(0.2,"Periodic"),(0.8,"Chaotic")]):
        sys=HeartEMSystem(w0=1.0,g1=gv)
        sol=sys.integrate(T=350); t=sol.t; mask=t>100
        axes[0,col].plot(t[mask][:2500],sol.y[2,mask][:2500],
                         lw=0.8,color=["#1A5276","#922B21"][col])
        axes[0,col].set_xlabel("Time $t$"); axes[0,col].set_ylabel("$x(t)$")
        axes[0,col].set_title(f"({'ab'[col]}) {label} ($g_1={gv}$)")
        xc,Ec=poincare_section(sys,T=350,transient=100)
        axes[1,col].scatter(xc,Ec,s=4,color=["#1A5276","#922B21"][col],alpha=0.8)
        axes[1,col].set_xlabel("$x$"); axes[1,col].set_ylabel("$E$")
        axes[1,col].set_title(f"({'cd'[col]}) Poincare section ($g_1={gv}$)")
    plt.suptitle("FIG. 5.  Time series and Poincare sections",fontsize=9,y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT,"fig5_timeseries.png"))
    plt.close()
    print("  -> fig5_timeseries.png")


FIGS = {1:fig1, 2:fig2, 3:fig3, 4:fig4, 5:fig5}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig",type=int,choices=[1,2,3,4,5])
    args = parser.parse_args()
    (FIGS[args.fig] if args.fig else [f() for f in FIGS.values()])
    print("Done.")
