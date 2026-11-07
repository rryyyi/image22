from __future__ import annotations

from pathlib import Path

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "report_assets"
ASSETS.mkdir(parents=True, exist_ok=True)
TABLES = ROOT / "results" / "tables"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.default"] = "bf"

EQUATIONS: list[tuple[str, str, str]] = [
    ("eq_01",
     r"I(x,y)=R(x,y)\cdot L(x,y),\qquad \hat{R}(x,y)=\frac{I(x,y)}{\max\,(L(x,y),\;\varepsilon_L)}",
     "（1）"),
    ("eq_02",
     r"A\bullet B=(A\oplus B)\ominus B,\qquad L=G_{\sigma}\left(A\bullet B\right),\quad \sigma=\max(1,\,w_B/8)",
     "（2）"),
    ("eq_03",
     r"L_{\mathrm{tgt}}=\mathrm{P}_{92}(L),\qquad g(x)=\mathrm{clip}\!\left(\frac{L_{\mathrm{tgt}}}{\max(L(x),0.025)},\;\frac{1}{M},\;M\right),\qquad \tilde{g}=1+(g-1)\,s",
     "（3）"),
    ("eq_04",
     r"Y'(x)=\mathrm{clip}\left(Y(x)\cdot\tilde{g}(x),\,0,\,1\right)",
     "（4）"),
    ("eq_05",
     r"\alpha(x)=s_c\cdot\mathrm{clip}\!\left(\frac{S_{\max}-S(x)}{0.25}\right)\cdot\mathrm{clip}\!\left(\frac{Y(x)-0.18}{0.25}\right),\qquad C'=(1-\alpha)\,I_Y+\alpha\,\mathrm{clip}(I\circ g_c,\,0,\,1)",
     "（5）"),
    ("eq_06",
     r"L_r=G_{35}(R_{\mathrm{gray}}),\quad r=\mathrm{clip}\!\left(\frac{\mathrm{P}_{92}(L_r)}{\max(L_r,0.08)},\,0.85,\,1.25\right),\quad R\leftarrow R\cdot\left(1+(r-1)\,s_r\right)",
     "（6）"),
    ("eq_07",
     r"a_k=\frac{\mathrm{cov}_k(I,p)}{\mathrm{var}_k(I)+\varepsilon},\qquad b_k=\bar{p}_k-a_k\bar{I}_k,\qquad q_i=a_k I_i+b_k",
     "（7）"),
    ("eq_08",
     r"w^{t+1}=\max\!\left(0,\;w^{t}+\eta\sum_{j\in\mathcal{N}}\min\left(s^{t}_{j}-s^{t},\,0\right)+e^{-8t/T}\left(s_{\max}-s^{t}\right)\right),\;\; s^{t}=w^{t}+A",
     "（8）"),
    ("eq_09",
     r"\mathrm{MSE}=\frac{1}{N}\sum_{x}\left(\hat{R}(x)-R_{\mathrm{gt}}(x)\right)^{2},\qquad \mathrm{PSNR}=10\log_{10}\frac{1}{\mathrm{MSE}}",
     "（9）"),
    ("eq_10",
     r"\mathrm{SSIM}(x,y)=\frac{\left(2\mu_x\mu_y+C_1\right)\left(2\sigma_{xy}+C_2\right)}{\left(\mu_x^{2}+\mu_y^{2}+C_1\right)\left(\sigma_x^{2}+\sigma_y^{2}+C_2\right)}",
     "（10）"),
    ("eq_11",
     r"M_s(x)=\mathbf{1}\left[G_{\sigma}(A\bullet G_{\mathrm{clean}})-G_{\sigma}(A\bullet G_{\mathrm{in}})>\tau\right]",
     "（11）"),
    ("eq_12",
     r"\mathbf{x}'\sim H\,\mathbf{x},\qquad w'=\mathbf{h}_3^{\top}\mathbf{x},\qquad (u',v')=\left(\frac{\mathbf{h}_1^{\top}\mathbf{x}}{w'},\;\frac{\mathbf{h}_2^{\top}\mathbf{x}}{w'}\right)",
     "（12）"),
    ("eq_13",
     r"\hat{\sigma}=\frac{\mathrm{median}\left(\left|\nabla I\right|_{\mathrm{paper}}\right)}{0.6745},\qquad Q=\sum_{k}w_k\,q_k,\;\; q_k\in\{\mathrm{sharp},\,\mathrm{contrast},\,\mathrm{uniform},\,\mathrm{noise},\,\mathrm{glare}\}",
     "（13）"),
    ("eq_14",
     r"E_r=E_{\mathrm{out}}/E_{\mathrm{in}},\quad E_r<0.72\Rightarrow\mathrm{warn};\qquad \rho_{\mathrm{glare}}>2.5\%\Rightarrow\mathrm{reject}",
     "（14）"),
]


def render_equations():
    for name, latex, number in EQUATIONS:
        fig = plt.figure(figsize=(17, 1.2), dpi=300)
        fig.patch.set_facecolor("white")
        try:
            fig.text(0.01, 0.5, f"${latex}$", fontsize=19, ha="left", va="center", color="#1A2733")
            fig.text(0.999, 0.5, number, fontsize=13, ha="right", va="center", color="#55606C", fontweight="bold")
            fig.savefig(ASSETS / f"{name}.png", dpi=300, facecolor="white",
                        bbox_inches="tight", pad_inches=0.10)
        except Exception as exc:  # noqa: BLE001
            print("FAILED", name, exc)
        plt.close(fig)
    print(f"equations rendered: {len(EQUATIONS)}")


def _style_ax(ax):
    ax.set_facecolor("#FBFCFE")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C9D2DC")
    ax.spines["bottom"].set_color("#C9D2DC")
    ax.tick_params(colors="#33404F", labelsize=10.5)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, color="#DCE4EC", alpha=0.9)
    ax.set_axisbelow(True)


def _read_csv(name):
    with (TABLES / name).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def make_per_image_delta():
    rows = [r for r in _read_csv("failure_ranking.csv")]
    rows.sort(key=lambda r: float(r["gain_over_nn"]), reverse=True)
    samples = [r["sample"] for r in rows]
    delta = [float(r["gain_over_nn"]) for r in rows]
    colors = ["#2F6FB0" if d >= 0 else "#C9693B" for d in delta]
    fig, ax = plt.subplots(figsize=(12.4, 4.4), dpi=180)
    fig.patch.set_facecolor("white")
    _style_ax(ax)
    ax.bar(samples, delta, color=colors, width=0.62, edgecolor="white", linewidth=0.6)
    ax.axhline(0, color="#8A97A5", linewidth=1.0)
    for i, d in enumerate(delta):
        ax.text(i, d + (0.28 if d >= 0 else -0.28), f"{d:+.1f}", ha="center",
                va="bottom" if d >= 0 else "top", fontsize=8.6, color="#23384E")
    ax.set_ylabel("自主方法 − DocShadow / dB", fontsize=11, color="#44515F")
    ax.set_title("20 对样本上自主方法相对 DocShadow 的逐图 PSNR 增益", fontsize=14.5, color="#1D3B5F", pad=10)
    ax.tick_params(axis="x", rotation=60, labelsize=9)
    fig.tight_layout()
    fig.savefig(ASSETS / "per_image_delta.jpg", dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def make_estimator_chart():
    order = ["morphological", "contrast_guided", "guided", "retinex", "water_filling"]
    labels = ["形态学闭运算", "对比度引导", "引导滤波", "Retinex", "注水（简化）"]
    rows = {r["method"]: r for r in _read_csv("estimator_summary.csv") if r["region"] == "full"}
    psnr = [float(rows[m]["psnr"]) for m in order]
    ssim = [float(rows[m]["ssim"]) for m in order]
    time_s = [float(rows[m]["runtime_seconds"]) for m in order]
    colors = ["#2F6FB0", "#7C93AC", "#7C93AC", "#A9B4C2", "#A9B4C2"]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), dpi=180)
    fig.patch.set_facecolor("white")
    for ax in axes:
        _style_ax(ax)
    bars = axes[0].bar(labels, psnr, color=colors, width=0.56, edgecolor="white", linewidth=0.6)
    axes[0].set_title("五种光照估计器的全图 PSNR", fontsize=14, color="#1D3B5F", pad=10)
    axes[0].set_ylabel("PSNR / dB", fontsize=11, color="#44515F")
    axes[0].set_ylim(14.5, 19.5)
    for bar, value in zip(bars, psnr):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.07, f"{value:.3f}",
                     ha="center", va="bottom", fontsize=9.6, color="#23384E")
    axes[0].tick_params(axis="x", labelsize=9.6)

    bars = axes[1].bar(labels, time_s, color=["#C9693B" if i == 0 else "#E3A25B" for i in range(5)],
                       width=0.56, edgecolor="white", linewidth=0.6)
    axes[1].set_title("单张平均耗时（同机同数据）", fontsize=14, color="#1D3B5F", pad=10)
    axes[1].set_ylabel("耗时 / s", fontsize=11, color="#44515F")
    for bar, value in zip(bars, time_s):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.008, f"{value:.3f}",
                     ha="center", va="bottom", fontsize=9.6, color="#23384E")
    axes[1].tick_params(axis="x", labelsize=9.6)
    fig.text(0.5, 0.01, "SSIM：形态学 0.8425 / 对比度引导 0.8421 / 引导滤波 0.8419 / Retinex 0.8360 / 注水 0.8175",
             ha="center", fontsize=9.5, color="#77828F")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(ASSETS / "estimator_chart.jpg", dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def make_ablation_chart():
    rows = {r["variant"]: r for r in _read_csv("ablation_summary.csv") if r["region"] == "full"}
    variants = ["full", "illumination_only", "no_color_correction", "no_residual"]
    labels = ["完整配置", "仅光照修复", "关闭彩色校正", "关闭残余校正"]
    psnr = [float(rows[v]["psnr"]) for v in variants]
    ssim = [float(rows[v]["ssim"]) for v in variants]
    rmse = [float(rows[v]["rmse"]) for v in variants]
    colors = ["#2F6FB0", "#2F6FB0", "#7C93AC", "#C9693B"]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4), dpi=180)
    fig.patch.set_facecolor("white")
    for ax in axes:
        _style_ax(ax)
    bars = axes[0].bar(labels, psnr, color=colors, width=0.52, edgecolor="white", linewidth=0.6)
    axes[0].set_title("消融配置的全图 PSNR", fontsize=14, color="#1D3B5F", pad=10)
    axes[0].set_ylabel("PSNR / dB", fontsize=11, color="#44515F")
    axes[0].set_ylim(18.0, 19.25)
    for bar, value in zip(bars, psnr):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}",
                     ha="center", va="bottom", fontsize=9.8, color="#23384E")
    axes[0].tick_params(axis="x", labelsize=10)

    x = np.arange(len(labels))
    width = 0.36
    axes[1].bar(x - width / 2, ssim, width, label="SSIM", color="#2F6FB0", edgecolor="white")
    axes[1].bar(x + width / 2, rmse, width, label="RMSE", color="#E3A25B", edgecolor="white")
    axes[1].set_xticks(x, labels, fontsize=10)
    axes[1].set_title("消融配置的 SSIM 与 RMSE", fontsize=14, color="#1D3B5F", pad=10)
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(frameon=False, fontsize=10, loc="upper right")
    for xi, value in zip(x - width / 2, ssim):
        axes[1].text(xi, value + 0.02, f"{value:.4f}", ha="center", fontsize=8.8, color="#23384E")
    for xi, value in zip(x + width / 2, rmse):
        axes[1].text(xi, value + 0.02, f"{value:.4f}", ha="center", fontsize=8.8, color="#23384E")
    fig.tight_layout()
    fig.savefig(ASSETS / "ablation_chart.jpg", dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def make_sensitivity_chart():
    rows = _read_csv("sensitivity_summary.csv")
    def curve(param):
        values, psnr = [], []
        seen = {}
        for r in rows:
            if r["parameter"] == param and r["region"] == "full":
                seen[float(r["value"])] = float(r["psnr"])
        for key in sorted(seen):
            values.append(key)
            psnr.append(seen[key])
        return values, psnr

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.2), dpi=180)
    fig.patch.set_facecolor("white")
    for ax in axes:
        _style_ax(ax)
        ax.grid(axis="x", linestyle="--", linewidth=0.7, color="#DCE4EC", alpha=0.9)

    values, psnr = curve("morph_kernel")
    axes[0].plot(values, psnr, "-o", color="#2F6FB0", linewidth=2.2, markersize=7)
    axes[0].set_title("形态学核大小 $w_B$ 的敏感性", fontsize=14, color="#1D3B5F", pad=10)
    axes[0].set_xlabel("核大小 $w_B$ / px", fontsize=11, color="#44515F")
    axes[0].set_ylabel("全图 PSNR / dB", fontsize=11, color="#44515F")
    axes[0].set_xticks(values)
    for v, p in zip(values, psnr):
        axes[0].annotate(f"{p:.3f}", (v, p), textcoords="offset points", xytext=(0, 9),
                         ha="center", fontsize=9.4, color="#23384E")

    values, psnr = curve("max_gain")
    axes[1].plot(values, psnr, "-o", color="#C9693B", linewidth=2.2, markersize=7)
    axes[1].set_title("最大增益 $M$ 的敏感性", fontsize=14, color="#1D3B5F", pad=10)
    axes[1].set_xlabel("最大增益 $M$", fontsize=11, color="#44515F")
    axes[1].set_xticks(values)
    for v, p in zip(values, psnr):
        axes[1].annotate(f"{p:.3f}", (v, p), textcoords="offset points", xytext=(0, 9),
                         ha="center", fontsize=9.4, color="#23384E")
    fig.tight_layout()
    fig.savefig(ASSETS / "sensitivity_chart.jpg", dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def make_gain_curve():
    target = 0.85
    L = np.linspace(0.06, 1.0, 400)
    M = 3.2
    g = np.clip(target / np.maximum(L, 0.025), 1.0 / M, M)
    g_tilde = 1.0 + (g - 1.0) * 0.92
    fig, ax = plt.subplots(figsize=(9.6, 4.6), dpi=180)
    fig.patch.set_facecolor("white")
    _style_ax(ax)
    ax.plot(L, g, "-", color="#7C93AC", linewidth=2.0, label=r"原始增益 $g=\mathrm{clip}(L_{\mathrm{tgt}}/L,\,1/M,\,M)$")
    ax.plot(L, g_tilde, "-", color="#2F6FB0", linewidth=2.6, label=r"混合后增益 $\tilde{g}=1+(g-1)\,s$，$s=0.92$")
    ax.axhline(M, color="#C9693B", linestyle="--", linewidth=1.1)
    ax.axhline(1.0 / M, color="#C9693B", linestyle="--", linewidth=1.1)
    ax.axhline(1.0, color="#8A97A5", linestyle=":", linewidth=1.0)
    ax.axvline(target, color="#2F6FB0", linestyle=":", linewidth=1.1)
    ax.text(0.015, M - 0.16, r"上限 $M=3.2$", fontsize=10, color="#C9693B")
    ax.text(0.015, 1.0 / M + 0.10, r"下限 $1/M$", fontsize=10, color="#C9693B")
    ax.text(target + 0.012, 0.42, r"$L_{\mathrm{tgt}}=\mathrm{P}_{92}(L)$", fontsize=10, color="#2F6FB0")
    ax.set_xlabel("估计照明场 $L(x)$", fontsize=11, color="#44515F")
    ax.set_ylabel("增益", fontsize=11, color="#44515F")
    ax.set_title("增益函数（3）的形状：限幅、分位数目标与强度混合", fontsize=14, color="#1D3B5F", pad=10)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    fig.tight_layout()
    fig.savefig(ASSETS / "gain_curve.jpg", dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def make_shadow_ratio():
    rows = _read_csv("failure_ranking.csv")
    ratios = np.array([float(r["shadow_ratio"]) for r in rows]) * 100
    fig, ax = plt.subplots(figsize=(9.6, 4.2), dpi=180)
    fig.patch.set_facecolor("white")
    _style_ax(ax)
    bins = np.arange(0, 105, 10)
    ax.hist(ratios, bins=bins, color="#2F6FB0", edgecolor="white", linewidth=0.8, alpha=0.92)
    ax.axvline(ratios.mean(), color="#C9693B", linestyle="--", linewidth=1.6)
    ax.text(ratios.mean() + 1.5, ax.get_ylim()[1] * 0.82, f"均值 {ratios.mean():.1f}%",
            fontsize=11, color="#C9693B")
    ax.set_xlabel("估计阴影区占比 / %", fontsize=11, color="#44515F")
    ax.set_ylabel("样本数", fontsize=11, color="#44515F")
    ax.set_title("20 对 Jung 数据的阴影区占比分布（掩膜由式（11）估计）", fontsize=14, color="#1D3B5F", pad=10)
    fig.tight_layout()
    fig.savefig(ASSETS / "shadow_ratio.jpg", dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main():
    render_equations()
    make_per_image_delta()
    make_estimator_chart()
    make_ablation_chart()
    make_sensitivity_chart()
    make_gain_curve()
    make_shadow_ratio()
    print("charts done")


if __name__ == "__main__":
    main()
