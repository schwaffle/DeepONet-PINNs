"""Build a self-contained HTML/Markdown report for PINN-Raissi vs M1 PI-DeepONet."""
from __future__ import annotations
import argparse, base64, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = ["M1_PI_DeepONet", "PINN_Raissi"]
TITLES = {"M1_PI_DeepONet": "M1 — PI-DeepONet", "PINN_Raissi": "PINN-Raissi"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def fig_fields(pred, out):
    nt = len(pred["plot_times"]); nx=len(pred["plot_x_axis"]); ny=len(pred["plot_y_axis"])
    truth=np.asarray(pred["plot_truth"]).reshape(nt,nx,ny)
    available=[m for m in METHODS if m in pred]
    if not available: return
    fields={m:np.asarray(pred[m]).reshape(nt,nx,ny) for m in available}
    rows=[("HF reference",None)]+[(TITLES[m],m) for m in available]
    its=sorted(set([max(0,nt//4),nt//2,min(nt-1,3*nt//4)]))
    arrays=[truth]+list(fields.values()); vmin=min(a.min() for a in arrays); vmax=max(a.max() for a in arrays)
    fig,axs=plt.subplots(len(rows),len(its),figsize=(4.1*len(its),3.2*len(rows)),squeeze=False)
    for r,(label,method) in enumerate(rows):
        for c,it in enumerate(its):
            f=truth[it] if method is None else fields[method][it]
            im=axs[r,c].imshow(f.T,origin="lower",extent=[0,1,0,1],aspect="equal",vmin=vmin,vmax=vmax)
            axs[r,c].set_xlabel("x"); axs[r,c].set_ylabel("y")
            if r==0: axs[r,c].set_title(f"t={pred['plot_times'][it]:.3f}")
            if c==0: axs[r,c].text(-0.22,0.5,label,rotation=90,transform=axs[r,c].transAxes,ha="center",va="center")
    fig.colorbar(im,ax=axs.ravel().tolist(),label="concentration u",shrink=.82)
    fig.suptitle("High-fidelity concentration fields",y=.995)
    fig.subplots_adjust(left=.10, right=.90, bottom=.08, top=.92, wspace=.24, hspace=.30)
    fig.savefig(out/"field_comparison.png",dpi=180); plt.close(fig)


def fig_errors(pred,out):
    nt=len(pred["plot_times"]); nx=len(pred["plot_x_axis"]); ny=len(pred["plot_y_axis"]); it=nt//2
    truth=np.asarray(pred["plot_truth"]).reshape(nt,nx,ny)[it]
    available=[m for m in METHODS if m in pred]
    if not available:return
    errors={m:np.abs(np.asarray(pred[m]).reshape(nt,nx,ny)[it]-truth) for m in available}
    vmax=max(float(e.max()) for e in errors.values())
    fig,axs=plt.subplots(1,len(errors),figsize=(5*len(errors),4),squeeze=False)
    for ax,(m,e) in zip(axs[0],errors.items()):
        im=ax.imshow(e.T,origin="lower",extent=[0,1,0,1],aspect="equal",vmin=0,vmax=vmax,cmap="magma")
        ax.set_title(TITLES[m]); ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(im,ax=axs.ravel().tolist(),label="absolute error")
    fig.suptitle(f"Absolute error maps at t={pred['plot_times'][it]:.3f}")
    fig.subplots_adjust(left=.10, right=.88, bottom=.12, top=.86, wspace=.30)
    fig.savefig(out/"absolute_error_maps.png",dpi=180); plt.close(fig)


def fig_training(out,figdir):
    histories={m:read_json(out/f"history_{m}.json") for m in METHODS}
    histories={m:h for m,h in histories.items() if h}
    if not histories:return
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for m,h in histories.items():
        it=[r["iteration"] for r in h]
        axs[0].semilogy(it,[max(r["loss"],1e-14) for r in h],label=TITLES[m])
        axs[1].semilogy(it,[max(r["pde"],1e-14) for r in h],label=TITLES[m])
    axs[0].set(xlabel="optimizer iteration",ylabel="total loss",title="Objective")
    axs[1].set(xlabel="optimizer iteration",ylabel="dimensionless PDE residual MSE",title="Physics residual")
    for ax in axs: ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(figdir/"training_curves.png",dpi=180); plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(13,3.8))
    for j,param in enumerate(("Q","xs","ys")):
        for m,h in histories.items(): axs[j].plot([r["iteration"] for r in h],[r[param] for r in h],label=TITLES[m])
        axs[j].set(xlabel="iteration",ylabel=param,title=f"Estimated {param}"); axs[j].grid(alpha=.25)
    axs[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(figdir/"parameter_convergence.png",dpi=180); plt.close(fig)


def fig_parameter_errors(df,out):
    if df.empty:return
    cols=["relerr_Q","relerr_xs","relerr_ys"]
    fig,ax=plt.subplots(figsize=(7,4)); x=np.arange(len(df)); w=.24
    for i,c in enumerate(cols): ax.bar(x+(i-1)*w,df[c].astype(float),w,label=c.replace("relerr_",""))
    ax.set_xticks(x,[TITLES.get(m,m) for m in df["method"]]); ax.set_ylabel("relative parameter error")
    ax.set_title("Source parameter identification"); ax.grid(axis="y",alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(out/"parameter_errors.png",dpi=180); plt.close(fig)


def embed(path): return "data:image/png;base64,"+base64.b64encode(path.read_bytes()).decode()


def build_report(out_dir, dataset=None, device="cpu"):
    out=Path(out_dir); figdir=out/"figures"; figdir.mkdir(parents=True,exist_ok=True)
    metrics_path=out/"results_seed.csv"; df=pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    pred_path=out/"predictions_plot.npz"
    if pred_path.exists():
        with np.load(pred_path) as raw: pred={k:raw[k] for k in raw.files}
        fig_fields(pred,figdir); fig_errors(pred,figdir)
    fig_parameter_errors(df,figdir); fig_training(out,figdir)
    cfg=read_json(out/"training_metadata.json")
    direct=read_json(out/"direct_hf_ls_reference.json")
    table=df.to_markdown(index=False,floatfmt=".5g") if len(df) else "No results_seed.csv found."
    lines=[
      "# Exp. 1 — PINN-Raissi vs M1 PI-DeepONet","",
      "## Research question","Compare a single-fidelity inverse PINN with composite multi-fidelity PI-DeepONet for identifying a Gaussian source in 2-D advection–diffusion.","",
      "## Methods","- PINN-Raissi: fully connected tanh network; trainable bounded source parameters; Adam then SciPy L-BFGS-B.",
      "- M1 PI-DeepONet: LF modified DeepONet + linear HF DeepONet + nonlinear HF modified DeepONet; HF output is the sum of the linear and nonlinear blocks.",
      "- MPINN-v2 is excluded from this comparison.","",
      "## Data and fidelity protocol","LF simulations use a 21×21 grid; HF simulations use a 129×129 grid. Both use central differences. The target source is held out of the LF parameter-function ensemble. HF inverse training uses 80 noisy observations at noise_fraction=0.005. Clean HF test points are taken from all saved HF grid-time points, excluding exact observation coordinate-time locations. A separate downsampled grid is used only for figures.","",
      "## Objective","For M1, the six terms in Howard et al. (2023), Eq. (15), are retained: physics, LF data, nonlinear-branch regularization, LF-branch regularization, initial condition and boundary condition. Because this application is an inverse problem with sparse noisy HF measurements, we add λ_obs L_HF-observation as a seventh, task-specific term. The same HF-observation loss is used for PINN-Raissi. This is an explicit extension of Eq. (15), not a claim that Eq. (15) contains the observation term.","",
      "## Training and leakage prevention","The target clean test values are not used in training, scaling, hyperparameter selection or early stopping. Concentration scaling is computed from noisy training observations only. LF train/validation is split by complete source realizations. The same HF observations, collocation points and boundary/initial sampling seeds are used for both methods within each seed. L-BFGS-B uses fixed collocation, BC, IC and LF mini-batch points throughout its closure evaluations.","",
      "## Metrics (one seed)",table,"",
      "`test_RMSE/MAE/relL2` use all clean, non-observed HF grid-time points, including analytically prescribed boundary and initial-condition points. `test_interior_*` excludes the four boundary sides and the t=0 plane, and is the more stringent field metric away from imposed constraints.","",
      "## Direct HF-LS control (not a neural method)",json.dumps(direct,indent=2,ensure_ascii=False) if direct else "Not run.","",
      "## Reproduction","```bash","python -m pip install -r requirements.txt",
      "python run_exp1.py --seed 7 --model both --device cpu", "```","",
      "## Current limitations","The HF model is the project's finite-difference solver, not WRF-Chem. WRF-Chem coupling, runtime calibration (the intended expensive 1/5 real-time HF cost), multi-seed statistics, cross-scheme (central vs upwind) comparison and physical validation against atmospheric chemistry data remain future work.","",
      "## Figures","- figures/field_comparison.png","- figures/absolute_error_maps.png","- figures/parameter_errors.png","- figures/training_curves.png","- figures/parameter_convergence.png",""
    ]
    (out/"report_exp1.md").write_text("\n".join(lines),encoding="utf-8")
    figs=[]
    for fn,cap in [("field_comparison.png","HF reference and method predictions."),("absolute_error_maps.png","Absolute concentration-field errors."),("parameter_errors.png","Relative errors in Q, xs and ys."),("training_curves.png","Training objective and PDE residual."),("parameter_convergence.png","Estimated source parameters during optimization.")]:
        p=figdir/fn
        if p.exists(): figs.append(f'<figure><img src="{embed(p)}"><figcaption>{cap}</figcaption></figure>')
    table_html=df.to_html(index=False,border=0,float_format=lambda v:f"{v:.5g}") if len(df) else "<p>No results available.</p>"
    html=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PINN vs PI-DeepONet</title><style>body{{font:16px/1.55 system-ui,sans-serif;max-width:1180px;margin:35px auto;padding:0 20px;color:#222}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #ddd;padding:7px;text-align:right}}th:first-child,td:first-child{{text-align:left}}figure{{margin:28px 0}}img{{max-width:100%;height:auto;border:1px solid #ddd}}figcaption{{color:#555;font-size:14px}}pre{{white-space:pre-wrap;background:#f5f5f5;padding:14px}}</style></head><body><h1>Exp. 1 — PINN-Raissi vs M1 PI-DeepONet</h1><h2>Configuration</h2><pre>{json.dumps(cfg,indent=2,ensure_ascii=False)}</pre><h2>Metrics</h2>{table_html}<h2>Figures</h2>{''.join(figs)}<h2>Method note</h2><p>Howard et al. Eq. (15) is retained, with a task-specific high-fidelity observation term added for inverse identification.</p></body></html>'''
    (out/"report_exp1.html").write_text(html,encoding="utf-8")
    return out/"report_exp1.html"


def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",default="results_exp1"); p.add_argument("--dataset",default=None); p.add_argument("--device",default="cpu")
    a=p.parse_args(); print(build_report(a.output,a.dataset,a.device))

if __name__=="__main__": main()
