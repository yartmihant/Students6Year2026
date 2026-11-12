"""Перенос по кольцу: upwind и Лакс–Вендрофф.

Зависимости: numpy, matplotlib.
Запуск: python 6.1_green_melnikov.py
Расчёт без графических окон: python 6.1_green_melnikov.py --no-gui
"""

import argparse
import numpy as np


c = 1.0
C = 0.8
SIZES = (50, 100, 200)
TIMES = (1.0, 5.0)
METHODS = ("Upwind", "Лакс–Вендрофф")


def initial(x):
    return 1 + 12 * np.sin(2 * np.pi * x)


def exact(x, t):
    return initial((x - c * t) % 1.0)


def upwind_step(u, courant):
    return u - courant * (u - np.roll(u, 1))


def lax_wendroff_step(u, courant):
    left, right = np.roll(u, 1), np.roll(u, -1)
    return (u - 0.5 * courant * (right - left)
            + 0.5 * courant**2 * (right - 2*u + left))


def compute(N):
    dx = 1.0 / N
    x = np.arange(N) * dx  # Точка x=1 не дублируется.
    dt_nominal = C * dx / c
    solutions = {name: initial(x) for name in METHODS}
    time_history = [0.0]
    mass_history = {name: [dx * np.sum(u)] for name, u in solutions.items()}
    snapshots = {}
    t = 0.0

    for target in TIMES:
        start = t
        n = int(np.ceil((target - start) / dt_nominal - 1e-12))
        for k in range(n):
            next_t = min(start + (k + 1) * dt_nominal, target)
            dt = next_t - t
            courant = c * dt / dx
            solutions[METHODS[0]] = upwind_step(solutions[METHODS[0]], courant)
            solutions[METHODS[1]] = lax_wendroff_step(solutions[METHODS[1]], courant)
            t = next_t
            time_history.append(t)
            for name in METHODS:
                mass_history[name].append(dx * np.sum(solutions[name]))
        # Сохраняем именно моменты t=1 и t=5, а не ближайшие временные слои.
        snapshots[target] = {name: u.copy() for name, u in solutions.items()}

    return {
        "x": x, "dx": dx, "snapshots": snapshots,
        "time": np.array(time_history),
        "mass": {name: np.array(values) for name, values in mass_history.items()},
    }


def error(result, target, name):
    u = result["snapshots"][target][name]
    return result["dx"] * np.sum(np.abs(u - exact(result["x"], target)))


def print_report(results):
    print("Перенос по кольцу: c=1, C=0.8; основной шаг dt=0.8*dx.")
    print("Последний шаг до t=1 и t=5 при необходимости сокращается.")
    print("Точное количество примеси M=1 во все моменты времени.\n")
    print(f"{'N':>5} {'t':>4} {'Метод':>17} {'E1':>13} {'M(t)':>17} {'max |M−M0|':>14}")
    for N, result in results.items():
        for target in TIMES:
            for name in METHODS:
                u = result["snapshots"][target][name]
                mass = result["dx"] * np.sum(u)
                history = result["mass"][name]
                selected = result["time"] <= target + 1e-13
                drift = np.max(np.abs(history[selected] - history[0]))
                print(f"{N:5d} {target:4.0f} {name:>17} {error(result, target, name):13.6e} "
                      f"{mass:17.12f} {drift:14.3e}")

    print("\nОтвет: сохранение количества примеси не означает правильного распределения.")
    print("Сумма значений может сохраняться при размытии профиля и ошибке его сдвига.")
    print("Здесь M остаётся близким к 1, но E1 не равна нулю и зависит от метода и сетки.")


def make_figures(plt, results):
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25})
    styles = {METHODS[0]: ("tab:blue", "--"), METHODS[1]: ("tab:orange", "-")}

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    fig.suptitle("Перенос гладкого профиля по кольцу, C=0.8")
    dense_x = np.linspace(0, 1, 1001)
    for row, target in enumerate(TIMES):
        for col, N in enumerate(SIZES):
            ax = axes[row, col]
            result = results[N]
            ax.plot(dense_x, exact(dense_x, target), "k", lw=2, label="Точное")
            for name in METHODS:
                color, style = styles[name]
                x = np.r_[result["x"], 1.0]
                u = result["snapshots"][target][name]
                ax.plot(x, np.r_[u, u[0]], style, color=color, label=name)
            # Замыкание линии при x=1 используется только для изображения.
            turns = "1 оборот" if target == 1 else "5 оборотов"
            ax.set_title(f"N={N}, t={target:g} ({turns})")
            ax.set_xlim(0, 1)
            if row == 1:
                ax.set_xlabel("x")
            if col == 0:
                ax.set_ylabel("u(x,t)")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    dxs = np.array([results[N]["dx"] for N in SIZES])
    for ax, target in zip(axes, TIMES):
        for name in METHODS:
            values = [error(results[N], target, name) for N in SIZES]
            ax.loglog(dxs, values, "o-", color=styles[name][0], label=name)
        ax.set(title=f"Ошибка после t={target:g}", xlabel="Шаг сетки dx", ylabel="E₁")
        ax.set_xticks(dxs, [f"{dx:g}" for dx in dxs])
        ax.minorticks_off()
        ax.invert_xaxis()
        ax.legend(fontsize=8)
    fig.tight_layout()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    fig.suptitle("Изменение количества примеси; M(0)=1")
    for ax, name in zip(axes, METHODS):
        for N in SIZES:
            result = results[N]
            mass = result["mass"][name]
            ax.plot(result["time"], mass - mass[0], label=f"N={N}")
        ax.set(title=name, xlabel="Время t", ylabel="M(t) − M(0)")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useOffset=False)
        ax.legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.92])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-gui", action="store_true", help="Расчёт и таблица без окон")
    args = parser.parse_args()
    results = {N: compute(N) for N in SIZES}
    print_report(results)
    if not args.no_gui:
        import matplotlib.pyplot as plt
        make_figures(plt, results)
        plt.show()


if __name__ == "__main__":
    main()
