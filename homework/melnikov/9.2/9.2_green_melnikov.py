"""Отражение импульса на стыке: линейные элементы и скоростной Верле.

Зависимости: numpy, matplotlib.
Запуск: python 9.2_green_melnikov.py
Без графических окон: python 9.2_green_melnikov.py --no-gui
"""

import argparse
import numpy as np


SIZES = (200, 400, 800)
TIMES = (0.5, 1.0, 1.5)
INTERFACE = 2.0
RHO_L, E_L = 1.0, 1.0
RHO_R, E_R = 1.0, 4.0
C_L, C_R = np.sqrt(E_L/RHO_L), np.sqrt(E_R/RHO_R)
Z_L, Z_R = RHO_L*C_L, RHO_R*C_R
R_EXACT = (Z_L-Z_R)/(Z_L+Z_R)
T_EXACT = 2*Z_L/(Z_L+Z_R)


def initial(x):
    a = 0.25
    mask = np.abs(x-1) < a
    angle = np.pi*(x[mask]-1)/(2*a)
    u, v = np.zeros_like(x), np.zeros_like(x)
    u[mask] = np.cos(angle)**4
    v[mask] = C_L*(2*np.pi/a)*np.cos(angle)**3*np.sin(angle)  # −c_L*g'(x)
    return u, v


def assemble(N):
    x = np.linspace(0, 4, N+1)
    h = 4.0/N
    if not np.isclose(x[N//2], INTERFACE, rtol=0, atol=1e-12):
        raise ValueError("Стык должен совпадать с узлом сетки.")
    mass_diag = np.zeros(N+1)
    mass_off = np.zeros(N)
    stiffness_diag = np.zeros(N+1)
    stiffness_off = np.zeros(N)
    for e in range(N):
        left_material = 0.5*(x[e]+x[e+1]) < INTERFACE
        rho, E = (RHO_L, E_L) if left_material else (RHO_R, E_R)
        Me = rho*h/6*np.array([[2.0, 1.0], [1.0, 2.0]])
        Ke = E/h*np.array([[1.0, -1.0], [-1.0, 1.0]])
        mass_diag[e:e+2] += np.diag(Me)
        mass_off[e] += Me[0, 1]
        stiffness_diag[e:e+2] += np.diag(Ke)
        stiffness_off[e] += Ke[0, 1]
    # Общий узел стыка получает вклады обоих материалов.
    # Строки массы суммируются до исключения закреплённых концов.
    mass = mass_diag.copy()
    mass[:-1] += mass_off
    mass[1:] += mass_off
    return x, mass[1:-1], stiffness_diag[1:-1], stiffness_off[1:-1]


def apply_stiffness(u, diagonal, off):
    result = diagonal*u
    result[:-1] += off*u[1:]
    result[1:] += off*u[:-1]
    return result


def compute(N):
    x, mass, diagonal, off = assemble(N)
    dt = 0.4*(4/N)/max(C_L, C_R)
    n = int(round(TIMES[-1]/dt))
    t = np.arange(n+1)*dt
    u0, v0 = initial(x)
    u, v = u0[1:-1].copy(), v0[1:-1].copy()
    acceleration = -apply_stiffness(u, diagonal, off)/mass
    U = np.zeros((n+1, N+1))
    U[0] = u0
    for i in range(n):
        half_velocity = v + 0.5*dt*acceleration
        u = u + dt*half_velocity
        new_acceleration = -apply_stiffness(u, diagonal, off)/mass
        v = half_velocity + 0.5*dt*new_acceleration
        acceleration = new_acceleration
        U[i+1, 1:-1] = u
    return {"x": x, "t": t, "dt": dt, "U": U, "final_velocity": v}


def time_index(result, target):
    i = int(round(target/result["dt"]))
    if not np.isclose(result["t"][i], target, rtol=0, atol=1e-12):
        raise ValueError("Момент наблюдения не попал на временную сетку.")
    return i


def measure(result):
    x = result["x"]
    u = result["U"][time_index(result, 1.5)]
    left = np.flatnonzero(x < INTERFACE)
    right = np.flatnonzero(x > INTERFACE)
    reflected = left[np.argmin(u[left])]  # Сохраняем отрицательный знак перемещения.
    transmitted = right[np.argmax(u[right])]
    return x[reflected], u[reflected], x[transmitted], u[transmitted]


def print_report(results):
    print("Стык x=2; слева rho=E=1, справа rho=1, E=4.")
    print("c_L=1, c_R=2; dt=0.4h/max(c_L,c_R). Начальная амплитуда равна 1.")
    print("Амплитуды и положения измеряются по узловым экстремумам при t=1.5.")
    print(f"Аналитические коэффициенты перемещения: R_u={R_EXACT:.8f}, T_u={T_EXACT:.8f}.")
    print("При t=1.5 ожидаемые центры импульсов: x_R=1.5 и x_T=3.\n")
    print(f"{'N':>5} {'dt':>10} {'x_R':>7} {'R_u изм.':>12} {'|ошибка R|':>12} "
          f"{'x_T':>7} {'T_u изм.':>12} {'|ошибка T|':>12}")
    for N, result in results.items():
        xr, ar, xt, at = measure(result)
        print(f"{N:5d} {result['dt']:10.6f} {xr:7.3f} {ar:12.7f} {abs(ar-R_EXACT):12.3e} "
              f"{xt:7.3f} {at:12.7f} {abs(at-T_EXACT):12.3e}")
    print("\nОтвет: справа импеданс Z_R=2 больше Z_L=1, поэтому R_u=−1/3:")
    print("отражённый импульс перемещения меняет знак.")
    print("Время прохождения импульса через стык одинаково для обеих сред,")
    print("но c_R=2c_L. Поэтому пространственная ширина прошедшего импульса")
    print("вдвое больше ширины падающего: 1 вместо 0.5.")


def make_figures(plt, results):
    from matplotlib.animation import FuncAnimation
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25})
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axvspan(0, 2, color="tab:blue", alpha=0.04)
    ax.axvspan(2, 4, color="tab:orange", alpha=0.06)
    ax.axvline(INTERFACE, color="0.4", ls=":", label="Стык материалов")
    lines = {}
    for N, color, style in [(200, "tab:orange", "--"), (800, "tab:blue", "-")]:
        lines[N], = ax.plot(results[N]["x"], results[N]["U"][0], style,
                            color=color, label=f"N={N}")
    ax.set(xlim=(0, 4), ylim=(-0.45, 1.1), xlabel="x", ylabel="Перемещение u")
    title = ax.set_title("Падающий импульс, t=0")
    ax.legend(loc="upper right")
    fig.tight_layout()
    frame_times = np.arange(76)*0.02

    def update(k):
        target = frame_times[k]
        for N, line in lines.items():
            line.set_ydata(results[N]["U"][time_index(results[N], target)])
        title.set_text(f"Отражение и прохождение, t={target:.2f}")
        return title, *lines.values()

    animation = FuncAnimation(fig, update, frames=len(frame_times), interval=40, repeat=False)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, target in zip(axes, TIMES):
        ax.axvline(INTERFACE, color="0.4", ls=":", label="Стык")
        for N in SIZES:
            result = results[N]
            ax.plot(result["x"], result["U"][time_index(result, target)], label=f"N={N}")
        ax.set(xlim=(0, 4), ylim=(-0.45, 1.1), xlabel="x", title=f"t={target:g}")
        ax.set_xticks(np.arange(5))
    axes[0].set_ylabel("Перемещение u")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return animation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-gui", action="store_true", help="Расчёт без графических окон")
    args = parser.parse_args()
    results = {N: compute(N) for N in SIZES}
    print_report(results)
    if not args.no_gui:
        import matplotlib.pyplot as plt
        animation = make_figures(plt, results)
        plt.show()


if __name__ == "__main__":
    main()
