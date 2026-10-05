import numpy as np
import matplotlib.pyplot as plt

def compute_mandelbrot(re_min=-2.0, re_max=1.0, im_min=-1.5, im_max=1.5,
                       width=600, height=600, max_iter=200, R=2.0):
    x = np.linspace(re_min, re_max, width)
    y = np.linspace(im_min, im_max, height)
    c = x[np.newaxis, :] + 1j * y[:, np.newaxis]
    
    z = np.zeros_like(c)
    iters = np.full(c.shape, max_iter, dtype=int)
    active = np.ones(c.shape, dtype=bool)
    R2 = R * R

    for i in range(1, max_iter + 1):
        z[active] = z[active]**2 + c[active]
        escaped = active & (np.real(z)**2 + np.imag(z)**2 > R2)
        iters[escaped] = i
        active[escaped] = False
        if not np.any(active):
            break

    return iters

def compute_julia(c_val, re_min=-2.0, re_max=2.0, im_min=-2.0, im_max=2.0,
                  width=600, height=600, max_iter=200):
    R = max(2.0, abs(c_val))
    R2 = R * R
    x = np.linspace(re_min, re_max, width)
    y = np.linspace(im_min, im_max, height)
    z = x[np.newaxis, :] + 1j * y[:, np.newaxis]
    
    iters = np.full(z.shape, max_iter, dtype=int)
    active = np.ones(z.shape, dtype=bool)

    for i in range(1, max_iter + 1):
        z[active] = z[active]**2 + c_val
        escaped = active & (np.real(z)**2 + np.imag(z)**2 > R2)
        iters[escaped] = i
        active[escaped] = False
        if not np.any(active):
            break

    return iters, R

def main():
    cmap = plt.cm.twilight_shifted.copy()
    cmap.set_bad(color='black')

    c_points = [
        (0.0, "c = 0", "lime"),
        (-1.0, "c = -1", "cyan"),
        (-0.123 + 0.745j, "c = -0.123 + 0.745i", "yellow"),
        (1.0, "c = 1", "magenta")
    ]

    m_re_min, m_re_max = -2.0, 1.0
    m_im_min, m_im_max = -1.5, 1.5
    m_iters = compute_mandelbrot(m_re_min, m_re_max, m_im_min, m_im_max,
                                 width=600, height=600, max_iter=200, R=2.0)

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 1, 1], wspace=0.25, hspace=0.3)

    ax_m = fig.add_subplot(gs[:, 0])
    m_masked = np.ma.masked_where(m_iters == 200, m_iters)
    im_m = ax_m.imshow(m_masked, extent=[m_re_min, m_re_max, m_im_min, m_im_max],
                       origin='lower', cmap=cmap)

    for c_val, label, col in c_points:
        ax_m.scatter([np.real(c_val)], [np.imag(c_val)], color=col, s=80,
                     edgecolors='black', linewidths=1.5, zorder=5, label=label)

    ax_m.set_aspect('equal')
    ax_m.set_title("Множество Мандельброта\n" + r"$z_0 = 0, \ c \in [-2, 1] \times [-1.5, 1.5], \ R=2, \ N_{\max}=200$", fontsize=12)
    ax_m.set_xlabel(r"$\mathrm{Re}(c)$", fontsize=11)
    ax_m.set_ylabel(r"$\mathrm{Im}(c)$", fontsize=11)
    ax_m.grid(True, linestyle="--", alpha=0.3)
    ax_m.legend(loc='lower left', fontsize=10, facecolor='#111111', labelcolor='white')

    cbar_m = fig.colorbar(im_m, ax=ax_m, orientation='horizontal', pad=0.08, fraction=0.046)
    cbar_m.set_label("Номер итерации выхода за R=2 (черный = внутри множества)", fontsize=10)

    julia_axes = [
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2])
    ]

    for idx, (c_val, label, col) in enumerate(c_points):
        ax_j = julia_axes[idx]
        j_iters, R_j = compute_julia(c_val, -2.0, 2.0, -2.0, 2.0,
                                     width=600, height=600, max_iter=200)
        j_masked = np.ma.masked_where(j_iters == 200, j_iters)
        im_j = ax_j.imshow(j_masked, extent=[-2.0, 2.0, -2.0, 2.0],
                           origin='lower', cmap=cmap)

        if c_val == 0.0:
            theta = np.linspace(0, 2 * np.pi, 300)
            ax_j.plot(np.cos(theta), np.sin(theta), color='cyan', linestyle='--',
                      linewidth=1.8, label=r"Окружность $|z|=1$")
            ax_j.legend(loc='upper right', fontsize=8, facecolor='#111111', labelcolor='white')

        ax_j.set_aspect('equal')
        ax_j.set_title(f"Заполненное множество Джулиа $K_c$\n{label}, $R = \\max(2, |c|) = {R_j:.1f}$",
                       fontsize=10, color='darkblue')
        ax_j.set_xlabel(r"$\mathrm{Re}(z_0)$", fontsize=9)
        ax_j.set_ylabel(r"$\mathrm{Im}(z_0)$", fontsize=9)
        ax_j.grid(True, linestyle="--", alpha=0.3)

        cbar_j = fig.colorbar(im_j, ax=ax_j, orientation='vertical', fraction=0.046, pad=0.04)
        cbar_j.set_label("Итерация выхода", fontsize=8)

    plt.suptitle(r"Комплексная динамика $z_{n+1} = z_n^2 + c$: Мандельброт и множества Джулиа",
                 fontsize=14, y=0.99)

    fig_c0, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    j0_iters, _ = compute_julia(0.0, -1.5, 1.5, -1.5, 1.5, width=600, height=600, max_iter=200)
    j0_masked = np.ma.masked_where(j0_iters == 200, j0_iters)

    im0 = ax1.imshow(j0_masked, extent=[-1.5, 1.5, -1.5, 1.5], origin='lower', cmap=cmap)
    theta = np.linspace(0, 2 * np.pi, 400)
    ax1.plot(np.cos(theta), np.sin(theta), color='cyan', linestyle='--',
             linewidth=2.2, label=r"Теоретическая граница $|z|=1$")
    ax1.set_aspect('equal')
    ax1.set_title(r"Численный расчет: множество Джулиа $K_0$" + "\n" + r"($c = 0$, сетка 600×600, $N_{\max}=200$)", fontsize=11)
    ax1.set_xlabel(r"$\mathrm{Re}(z_0)$", fontsize=10)
    ax1.set_ylabel(r"$\mathrm{Im}(z_0)$", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(loc='upper right', facecolor='#111111', labelcolor='white')
    fig_c0.colorbar(im0, ax=ax1, fraction=0.046, pad=0.04, label=r"Номер итерации выхода (черный — $|z_0| \leq 1$)")

    disk = plt.Circle((0, 0), 1.0, color='black', fill=True, label=r"Единичный диск $|z| \leq 1$ ($K_0$)")
    ax2.add_patch(disk)
    ax2.plot(np.cos(theta), np.sin(theta), color='red', linestyle='-',
             linewidth=2.2, label=r"Единичная окружность $|z|=1$ ($J_0$)")
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal')
    ax2.set_title(r"Теория: единичный диск и окружность" + "\n" + r"$z_{n+1} = z_n^2 \ \to \ |z_n| = |z_0|^{2^n}$", fontsize=11)
    ax2.set_xlabel(r"$\mathrm{Re}(z_0)$", fontsize=10)
    ax2.set_ylabel(r"$\mathrm{Im}(z_0)$", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(loc='upper right')
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
