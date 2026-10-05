import numpy as np
import matplotlib.pyplot as plt


G = 9.81
M1 = 1.0
M2 = 1.0
L1 = 1.0
L2 = 1.0
T_END = 20.0
STEPS = (0.04, 0.02, 0.01, 0.005)

# Оба угла отсчитываются от вертикали вниз, в радианах.
# Состояние: theta1, theta2, omega1, omega2.
STATE_0 = np.array([0.1, 0.05, 0.0, 0.0])


def rhs(state):
    # Два точечных груза на невесомых стержнях, без трения.
    theta1, theta2, omega1, omega2 = state
    delta = theta1 - theta2
    c = np.cos(delta)
    s = np.sin(delta)

    # Уравнения для угловых ускорений:
    # a11 * alpha1 + a12 * alpha2 = b1
    # a21 * alpha1 + a22 * alpha2 = b2
    a11 = (M1 + M2) * L1
    a12 = M2 * L2 * c
    a21 = L1 * c
    a22 = L2
    b1 = -(M1 + M2) * G * np.sin(theta1) - M2 * L2 * omega2**2 * s
    b2 = -G * np.sin(theta2) + L1 * omega1**2 * s

    determinant = a11 * a22 - a12 * a21
    alpha1 = (b1 * a22 - a12 * b2) / determinant
    alpha2 = (a11 * b2 - b1 * a21) / determinant
    return np.array([omega1, omega2, alpha1, alpha2])


def rk4_step(state, h):
    k1 = rhs(state)
    k2 = rhs(state + h * k1 / 2)
    k3 = rhs(state + h * k2 / 2)
    k4 = rhs(state + h * k3)
    return state + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def integrate(method, h):
    if h <= 0 or T_END <= 0 or min(M1, M2, L1, L2, G) <= 0:
        raise ValueError("Шаг, время и физические параметры должны быть положительными.")
    n = int(round(T_END / h))
    if n < 1 or not np.isclose(n * h, T_END, rtol=1e-12, atol=1e-12):
        raise ValueError("Выберите шаг, на который T_END делится без остатка.")

    # AB2 используется на равномерной сетке, без укороченного последнего шага.
    t = np.arange(n + 1) * h
    states = np.empty((n + 1, 4))
    states[0] = STATE_0

    if method == "RK4":
        for i in range(n):
            states[i + 1] = rk4_step(states[i], h)
    elif method == "AB2":
        # Для запуска многошагового метода первый шаг делаем методом RK4.
        states[1] = rk4_step(states[0], h)
        f_previous = rhs(states[0])
        for i in range(1, n):
            f_current = rhs(states[i])
            states[i + 1] = states[i] + h * (3 * f_current - f_previous) / 2
            f_previous = f_current
    else:
        raise ValueError("Метод должен быть RK4 или AB2.")
    return t, states


def lower_mass(states):
    theta1 = states[:, 0]
    theta2 = states[:, 1]
    x = L1 * np.sin(theta1) + L2 * np.sin(theta2)
    y = -L1 * np.cos(theta1) - L2 * np.cos(theta2)
    return x, y


def total_energy(states):
    theta1, theta2, omega1, omega2 = states.T
    kinetic = (
        (M1 + M2) * L1**2 * omega1**2 / 2
        + M2 * L2**2 * omega2**2 / 2
        + M2 * L1 * L2 * omega1 * omega2 * np.cos(theta1 - theta2)
    )
    # Нуль потенциальной энергии выбран в нижнем положении равновесия.
    potential = (
        (M1 + M2) * G * L1 * (1 - np.cos(theta1))
        + M2 * G * L2 * (1 - np.cos(theta2))
    )
    return kinetic + potential


def calculate():
    results = []
    for h in STEPS:
        t, rk4 = integrate("RK4", h)
        _, ab2 = integrate("AB2", h)
        results.append((h, t, {"RK4": rk4, "AB2": ab2}))
    return results


def plot_results(results):
    colors = {"RK4": "#0072b2", "AB2": "#d55e00"}
    styles = {"RK4": "-", "AB2": "--"}

    fig, axes = plt.subplots(len(results), 2, figsize=(13, 3 * len(results)),
                             squeeze=False, layout="constrained")
    fig.suptitle("Двойной маятник: изменение обоих углов")
    for row, (h, t, trajectories) in enumerate(results):
        for col in range(2):
            ax = axes[row, col]
            for name, states in trajectories.items():
                ax.plot(t, states[:, col], styles[name], color=colors[name], label=name)
            ax.set(title=f"Угол {col + 1}, h = {h:g} с", xlabel="t, с",
                   ylabel=f"θ{col + 1}, рад")
            ax.grid(alpha=0.3)
            ax.legend()

    rows = (len(results) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(12, 4 * rows),
                             squeeze=False, layout="constrained")
    fig.suptitle("Траектория нижнего груза (масштабы осей различаются)")
    for ax, (h, t, trajectories) in zip(axes.flat, results):
        for name, states in trajectories.items():
            x, y = lower_mass(states)
            ax.plot(x, y, styles[name], color=colors[name], label=name, linewidth=1)
        x0, y0 = lower_mass(STATE_0[None, :])
        ax.scatter(x0, y0, color="black", s=20, label="Начало", zorder=5)
        ax.set(title=f"h = {h:g} с", xlabel="x, м", ylabel="y, м")
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    for ax in list(axes.flat)[len(results):]:
        ax.set_visible(False)

    # Слева энергия, справа её относительное отклонение от начального значения.
    e0 = total_energy(STATE_0[None, :])[0]
    fig, axes = plt.subplots(len(results), 2, figsize=(13, 3 * len(results)),
                             squeeze=False, layout="constrained")
    fig.suptitle("Полная механическая энергия")
    for row, (h, t, trajectories) in enumerate(results):
        for name, states in trajectories.items():
            e = total_energy(states)
            relative_error = (e - e0) / e0 if e0 != 0 else np.zeros_like(e)
            axes[row, 0].plot(t, e, styles[name], color=colors[name], label=name)
            axes[row, 1].plot(t, relative_error, styles[name], color=colors[name], label=name)
        axes[row, 0].axhline(e0, color="black", linestyle=":", label="Начальная энергия")
        axes[row, 1].axhline(0, color="gray", linewidth=0.7)
        axes[row, 0].set(title=f"E(t), h = {h:g} с", ylabel="E, Дж")
        axes[row, 1].set(title=f"Отклонение энергии, h = {h:g} с", ylabel="(E - E0) / E0")
        for ax in axes[row]:
            ax.set_xlabel("t, с")
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useOffset=False)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)


def compare_results(results):
    e0 = total_energy(STATE_0[None, :])[0]
    print(f"\nНачальная энергия: {e0:.8f} Дж")
    print("Метод      h, с       max|E - E0|, Дж       max|E - E0| / E0")
    for h, t, trajectories in results:
        for name, states in trajectories.items():
            error = np.max(np.abs(total_energy(states) - e0))
            relative = error / e0 if e0 != 0 else 0.0
            print(f"{name:<6}   {h:7.4f}       {error:12.5e}         {relative:12.5e}")

    print("\nМаксимальное расхождение углов между RK4 и AB2:")
    print("h, с           Угол 1, рад       Угол 2, рад")
    for h, t, trajectories in results:
        difference = np.max(np.abs(trajectories["RK4"][:, :2] - trajectories["AB2"][:, :2]), axis=0)
        print(f"{h:7.4f}        {difference[0]:12.5e}      {difference[1]:12.5e}")
    print("Это сравнение двух численных решений, а не ошибка относительно точного решения.")


if __name__ == "__main__":
    results = calculate()
    plot_results(results)
    compare_results(results)
    plt.show()
