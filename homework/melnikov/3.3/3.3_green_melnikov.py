import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


grid_size = 600
Nmax = 200
parameters = [0j, -1 + 0j, -0.123 + 0.745j, 1 + 0j]
parameter_labels = ["0", "−1", "−0.123 + 0.745i", "1"]


def complex_grid(xmin, xmax, ymin, ymax):
    x = np.linspace(xmin, xmax, grid_size)
    y = np.linspace(ymin, ymax, grid_size)
    X, Y = np.meshgrid(x, y)
    return X + 1j * Y


def escape_iterations(z0, c, R):
    z = z0.copy()
    c_grid = np.zeros_like(z) + c
    active = np.ones(z.shape, dtype=bool)
    # Nmax + 1 обозначает отсутствие выхода, а не номер итерации.
    result = np.full(z.shape, Nmax + 1, dtype=int)

    # Для Джулиа начальная точка уже может лежать за радиусом R.
    escaped = np.abs(z) > R
    result[escaped] = 0
    active[escaped] = False

    for n in range(1, Nmax + 1):
        z[active] = z[active]**2 + c_grid[active]
        escaped = active & (np.abs(z) > R)
        result[escaped] = n
        active[escaped] = False
        if not np.any(active):
            break

    return result


# Цвет задаёт первую итерацию выхода; чёрный — выхода пока нет.
cmap = plt.get_cmap("turbo").copy()
cmap.set_over("black")
no_escape = Patch(facecolor="black", label=f"Нет выхода за {Nmax} итераций")


def draw_escape(ax, result, extent, title, real_label, imag_label):
    # Шкала каждого изображения охватывает найденные номера выхода.
    last_exit = max(1, np.max(result[result <= Nmax]))
    image = ax.imshow(
        result, origin="lower", extent=extent, aspect="equal",
        cmap=cmap, vmin=0, vmax=last_exit, interpolation="nearest"
    )
    ax.set_title(title)
    ax.set_xlabel(real_label)
    ax.set_ylabel(imag_label)
    return image


# Мандельброт: c меняется по сетке, z0 = 0 для всех точек.
mandelbrot_extent = [-2, 1, -1.5, 1.5]
c_values = complex_grid(*mandelbrot_extent)
mandelbrot = escape_iterations(np.zeros_like(c_values), c_values, R=2)

fig1, ax1 = plt.subplots(figsize=(8, 7))
image = draw_escape(
    ax1, mandelbrot, mandelbrot_extent,
    f"Множество Мандельброта: {grid_size}×{grid_size}, Nmax={Nmax}, R=2",
    "Re c", "Im c"
)
colorbar = fig1.colorbar(image, ax=ax1, shrink=0.85)
colorbar.set_label("Первая итерация выхода: |z| > R")

label_offsets = [(12, -22), (-48, -22), (12, 12), (-65, 12)]
for c, label, offset in zip(parameters, parameter_labels, label_offsets):
    ax1.plot(c.real, c.imag, "o", color="gold", markeredgecolor="black", clip_on=False)
    ax1.annotate(
        f"c = {label}", (c.real, c.imag), xytext=offset,
        textcoords="offset points", color="white", fontsize=9,
        bbox=dict(facecolor="black", alpha=0.7, edgecolor="none", pad=2)
    )
ax1.set_xlim(-2, 1)
ax1.set_ylim(-1.5, 1.5)
fig1.legend(handles=[no_escape], loc="lower center")
fig1.tight_layout(rect=[0, 0.06, 1, 1])


# Джулиа: z0 меняется по сетке, параметр c фиксирован.
julia_extent = [-2, 2, -2, 2]
z_values = complex_grid(*julia_extent)
julia_results = []

fig2, ax2 = plt.subplots(2, 2, figsize=(11, 10), layout="constrained")
fig2.suptitle(f"Заполненные множества Джулиа: сетка {grid_size}×{grid_size}, Nmax={Nmax}")

print("Параметр c          Точек без выхода за Nmax итераций")
for j, c in enumerate(parameters):
    R = max(2, abs(c))
    result = escape_iterations(z_values, c, R)
    julia_results.append(result)
    image = draw_escape(
        ax2[j // 2, j % 2], result, julia_extent,
        f"c = {parameter_labels[j]}, R={R:g}", "Re z₀", "Im z₀"
    )
    colorbar = fig2.colorbar(image, ax=ax2[j // 2, j % 2], shrink=0.8)
    colorbar.set_label("Первая итерация выхода")
    print(f"{parameter_labels[j]:18s} {np.count_nonzero(result == Nmax + 1)}")

fig2.supxlabel(f"Выход: |z| > R. Чёрный: нет выхода за {Nmax} итераций.\nИтерация 0: уже |z₀| > R.")


# При c = 0: z_n = z0^(2^n), поэтому K0 — замкнутый единичный диск.
# Его граница J0 — единичная окружность, а не весь заполненный диск.
numerical_disk = julia_results[0] == Nmax + 1
exact_disk = np.abs(z_values) <= 1
angle = np.linspace(0, 2 * np.pi, 1000)

fig3, ax3 = plt.subplots(1, 2, figsize=(11, 5))
fig3.suptitle(f"Джулиа при c = 0: сетка {grid_size}×{grid_size}, Nmax={Nmax}")
disk_images = [numerical_disk, exact_disk]
disk_titles = ["Численно: точки без выхода", "Точно: единичный диск |z₀| ≤ 1"]

for j in range(2):
    ax3[j].imshow(
        disk_images[j], origin="lower", extent=julia_extent,
        cmap="gray_r", vmin=0, vmax=1, aspect="equal", interpolation="nearest"
    )
    ax3[j].plot(np.cos(angle), np.sin(angle), color="red", linewidth=1,
                label="Единичная окружность |z₀| = 1")
    ax3[j].set_title(disk_titles[j])
    ax3[j].set_xlabel("Re z₀")
    ax3[j].set_ylabel("Im z₀")
    ax3[j].legend(loc="upper right", fontsize=8)
fig3.tight_layout()

different_points = np.count_nonzero(numerical_disk != exact_disk)
print(f"\nc = 0: несовпадений с единичным диском на сетке — {different_points}.")
print("При c = 0 заполненное множество — |z₀| ≤ 1, его граница — |z₀| = 1.")

# Конечный расчёт не доказывает ограниченность всей бесконечной орбиты.
print(f"\nЕсли точка не вышла за радиус за {Nmax} шагов, известно только, что")
print("выхода не было на проверенном интервале. Она может выйти позже.")
print("Это кандидат на принадлежность множеству, но не доказательство.")
print("Если выход произошёл, точка точно не принадлежит соответствующему множеству.")
if not np.any(julia_results[3] == Nmax + 1):
    print("Для c = 1 отсутствие чёрных точек на этой сетке не означает, что Kc пусто.")

plt.show()
