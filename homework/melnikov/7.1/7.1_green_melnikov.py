"""Остывание стержня: FTCS и неявный Эйлер при dt=0.4*dx**2.

Зависимости: numpy, matplotlib.
Запуск: python 7.1_green_melnikov.py
Расчёт без графических окон: python 7.1_green_melnikov.py --no-gui
"""

import argparse
import numpy as np


R = 0.4
SIZES = (25, 50, 100)  # Количество интервалов, узлов на один больше.
TIMES = (0.001, 0.01, 0.1)
METHODS = ("FTCS", "Неявный Эйлер")


def exact(x, t):
    u = (np.exp(-np.pi**2 * t) * np.sin(np.pi * x)
         + 0.3 * np.exp(-25 * np.pi**2 * t) * np.sin(5 * np.pi * x))
    u[0] = u[-1] = 0.0
    return u


def ftcs_step(u, ratio):
    new = np.zeros_like(u)
    new[1:-1] = u[1:-1] + ratio * (u[:-2] - 2*u[1:-1] + u[2:])
    return new


def factor_tridiagonal(size, ratio):
    # Диагональ 1+2r, соседние диагонали −r. Метод прогонки.
    diagonal = np.empty(size)
    diagonal[0] = 1 + 2 * ratio
    for i in range(1, size):
        diagonal[i] = 1 + 2 * ratio - ratio**2 / diagonal[i - 1]
    return diagonal


def implicit_step(u, ratio, diagonal):
    rhs = u[1:-1].copy()
    for i in range(1, len(rhs)):
        rhs[i] += ratio * rhs[i - 1] / diagonal[i - 1]
    rhs[-1] /= diagonal[-1]
    for i in range(len(rhs) - 2, -1, -1):
        rhs[i] = (rhs[i] + ratio * rhs[i + 1]) / diagonal[i]
    new = np.zeros_like(u)
    new[1:-1] = rhs
    return new


def compute(N, target):
    dx = 1.0 / N
    x = np.linspace(0, 1, N + 1)
    dt_nominal = R * dx**2
    n = int(np.ceil(target / dt_nominal - 1e-12))
    u_explicit = exact(x, 0.0)
    u_implicit = u_explicit.copy()
    normal_diagonal = factor_tridiagonal(N - 1, R)
    t = 0.0
    for k in range(n):
        next_t = min((k + 1) * dt_nominal, target)
        dt = next_t - t
        ratio = dt / dx**2
        if abs(ratio - R) < 1e-12:
            diagonal = normal_diagonal
        else:
            diagonal = factor_tridiagonal(N - 1, ratio)
        u_explicit = ftcs_step(u_explicit, ratio)
        u_implicit = implicit_step(u_implicit, ratio, diagonal)
        t = next_t

    solutions = {METHODS[0]: u_explicit, METHODS[1]: u_implicit}
    truth = exact(x, target)
    return {
        "x": x, "dx": dx, "dt": dt_nominal, "time": t,
        "solutions": solutions,
        "errors": {name: np.max(np.abs(u - truth)) for name, u in solutions.items()},
    }


def print_report(results):
    print("Остывание стержня: r=0.4; основной шаг dt=0.4*dx².")
    print("При удвоении числа интервалов dx уменьшается вдвое, dt — вчетверо.")
    print("Последний шаг до каждого заданного момента при необходимости сокращается.")
    print("Порядок p=log2(E_N/E_(2N)); p указан относительно предыдущей сетки.\n")
    print(f"{'t':>7} {'Метод':>16} {'N':>5} {'dx':>9} {'dt':>12} {'E_inf':>13} {'p':>7}")
    for target in TIMES:
        for name in METHODS:
            previous_error = None
            for N in SIZES:
                result = results[N, target]
                error = result["errors"][name]
                order = "—" if previous_error is None else f"{np.log2(previous_error / error):.3f}"
                print(f"{target:7.3f} {name:>16} {N:5d} {result['dx']:9.5f} "
                      f"{result['dt']:12.5e} {error:13.6e} {order:>7}")
                previous_error = error

    print("\nОбе схемы имеют погрешность O(dt + dx²). При dt=0.4*dx² ожидается порядок 2 по dx.")
    print("На грубых сетках оценки отличаются от 2; на мелких приближаются к 2.")
    print("Ответ: мелкие детали соответствуют большим пространственным частотам.")
    print("Мода sin(kπx) затухает как exp(−k²π²t), поэтому пятая мода затухает")
    print("в 25 раз быстрее первой по показателю экспоненты: её характерное время в 25 раз меньше.")


def make_figures(plt, results):
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25})
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True)
    fig.suptitle("Остывание стержня: FTCS и неявный Эйлер, r=0.4")
    dense_x = np.linspace(0, 1, 1001)
    for row, target in enumerate(TIMES):
        for col, N in enumerate(SIZES):
            ax = axes[row, col]
            result = results[N, target]
            ax.plot(dense_x, exact(dense_x, target), "k", lw=2, label="Точное")
            ax.plot(result["x"], result["solutions"][METHODS[0]], "--",
                    color="tab:blue", label=METHODS[0])
            ax.plot(result["x"], result["solutions"][METHODS[1]], ":",
                    color="tab:orange", lw=2, label=METHODS[1])
            ax.set(title=f"t={target:g}, N={N}", xlim=(0, 1))
            # В пределах строки сохраняем одинаковый масштаб температур.
            ax.set_ylim(0, 1.05 * np.max(exact(dense_x, target)))
            if row == 2:
                ax.set_xlabel("x")
            if col == 0:
                ax.set_ylabel("u(x,t)")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    dxs = np.array([1.0 / N for N in SIZES])
    for ax, target in zip(axes, TIMES):
        for name, color in zip(METHODS, ["tab:blue", "tab:orange"]):
            errors = np.array([results[N, target]["errors"][name] for N in SIZES])
            ax.loglog(dxs, errors, "o-", color=color, label=name)
        ax.set(title=f"t={target:g}", xlabel="Шаг сетки dx", ylabel="E∞")
        ax.set_xticks(dxs, [f"{dx:g}" for dx in dxs])
        ax.minorticks_off()
        ax.invert_xaxis()
        ax.legend(fontsize=8)
    fig.tight_layout()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-gui", action="store_true", help="Расчёт и таблица без окон")
    args = parser.parse_args()
    results = {(N, target): compute(N, target) for target in TIMES for N in SIZES}
    print_report(results)
    if not args.no_gui:
        import matplotlib.pyplot as plt
        make_figures(plt, results)
        plt.show()


if __name__ == "__main__":
    main()
