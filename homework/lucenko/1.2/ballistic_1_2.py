import numpy as np
import matplotlib.pyplot as plt


G = 9.81
M = 1.0
SPEED_0 = 20.0
ANGLE = np.radians(45)
STATE_0 = np.array([0.0, 0.0,
                    SPEED_0 * np.cos(ANGLE), SPEED_0 * np.sin(ANGLE)])

TAU = 2.0
STEPS = (0.2, 0.1, 0.05, 0.025)
TAU_VALUES = (0.5, 1.0, 2.0, 5.0, np.inf)


def rhs(state, tau):
    # Состояние: x, y, vx, vy. Ось y направлена вверх.
    x, y, vx, vy = state
    return np.array([vx, vy, -vx / tau, -G - vy / tau])


def euler_step(state, h, tau):
    return state + h * rhs(state, tau)


def rk4_step(state, h, tau):
    k1 = rhs(state, tau)
    k2 = rhs(state + h * k1 / 2, tau)
    k3 = rhs(state + h * k2 / 2, tau)
    k4 = rhs(state + h * k3, tau)
    return state + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def integrate(step, h, t_end, tau):
    if h <= 0 or t_end <= 0 or tau <= 0:
        raise ValueError("Шаг, время и tau должны быть положительными.")
    n = int(np.ceil(t_end / h))
    t = np.minimum(np.arange(n + 1) * h, t_end)
    t[-1] = t_end
    states = np.empty((len(t), 4))
    states[0] = STATE_0
    for i in range(len(t) - 1):
        states[i + 1] = step(states[i], t[i + 1] - t[i], tau)
    return t, states


def exact_solution(t, tau):
    t = np.atleast_1d(np.asarray(t, dtype=float))
    x0, y0, vx0, vy0 = STATE_0
    if np.isinf(tau):
        # Отсутствие сопротивления: обычная парабола.
        x = x0 + vx0 * t
        y = y0 + vy0 * t - G * t**2 / 2
        vx = np.full_like(t, vx0)
        vy = vy0 - G * t
    else:
        q = np.exp(-t / tau)
        a = -np.expm1(-t / tau)  # 1 - exp(-t/tau)
        x = x0 + vx0 * tau * a
        y = y0 + (vy0 + G * tau) * tau * a - G * tau * t
        vx = vx0 * q
        vy = (vy0 + G * tau) * q - G * tau
    return np.column_stack((x, y, vx, vy))


def flight_time(tau):
    # Ищем положительный корень y(t)=0 после вершины траектории.
    # Здесь предполагается запуск вверх с y0 >= 0.
    vy0 = STATE_0[3]
    if tau <= 0 or STATE_0[1] < 0 or vy0 <= 0:
        raise ValueError("Нужны tau > 0, y0 >= 0 и vy0 > 0.")
    left = vy0 / G if np.isinf(tau) else tau * np.log1p(vy0 / (G * tau))
    right = max(2 * left, 1.0)
    while exact_solution(right, tau)[0, 1] > 0:
        right *= 2
    for _ in range(60):
        middle = (left + right) / 2
        if exact_solution(middle, tau)[0, 1] > 0:
            left = middle
        else:
            right = middle
    return (left + right) / 2


def energy(states):
    potential = M * G * states[:, 1]
    kinetic = M * (states[:, 2]**2 + states[:, 3]**2) / 2
    return potential, kinetic, potential + kinetic


def calculate():
    # Для честного сравнения оба метода считаем до одного точного времени падения.
    # Их конечная высота может отличаться от нуля из-за численной ошибки.
    t_end = flight_time(TAU)
    results = []
    for h in STEPS:
        trajectories = {}
        for name, step in (("Эйлер", euler_step), ("RK4", rk4_step)):
            t, states = integrate(step, h, t_end, TAU)
            trajectories[name] = states
        results.append((h, t, exact_solution(t, TAU), trajectories))
    return results


def plot_results(results):
    colors = {"Эйлер": "#d55e00", "RK4": "#0072b2"}
    fine_t = np.linspace(0, flight_time(TAU), 1001)
    exact = exact_solution(fine_t, TAU)
    rows = (len(results) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(12, 4 * rows),
                             squeeze=False, layout="constrained")
    fig.suptitle(f"Баллистическое движение: tau = {TAU:g} с")
    for ax, (h, t, reference, trajectories) in zip(axes.flat, results):
        for name, states in trajectories.items():
            ax.plot(states[:, 0], states[:, 1], label=name, color=colors[name])
        ax.plot(exact[:, 0], exact[:, 1], "k--", label="Точное решение")
        ax.axhline(0, color="gray", linewidth=0.7)
        ax.set(title=f"h = {h:g} с", xlabel="x, м", ylabel="y, м")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)
        ax.legend()
    for ax in list(axes.flat)[len(results):]:
        ax.set_visible(False)

    exact_energies = energy(exact)
    fig, axes = plt.subplots(len(results), 3, figsize=(15, 3 * len(results)),
                             squeeze=False, layout="constrained")
    fig.suptitle(f"Энергии при линейном сопротивлении, tau = {TAU:g} с")
    for row, (h, t, reference, trajectories) in enumerate(results):
        for col, title in enumerate(("Потенциальная U", "Кинетическая K", "Полная E")):
            ax = axes[row, col]
            for name, states in trajectories.items():
                ax.plot(t, energy(states)[col], label=name, color=colors[name])
            ax.plot(fine_t, exact_energies[col], "k--", label="Точное решение")
            ax.set(title=f"{title}, h = {h:g} с", xlabel="t, с", ylabel="Энергия, Дж")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)


def compare_errors(results):
    print(f"\nТочное время падения при tau = {TAU:g} с: {flight_time(TAU):.6f} с")
    print("Оба метода сравниваются в одинаковые моменты времени.")
    print("Метод        h        max|r - r_exact|, м    max|E - E_exact|, Дж")
    for h, t, exact, trajectories in results:
        exact_energy = energy(exact)[2]
        for name, states in trajectories.items():
            error_r = np.max(np.linalg.norm(states[:, :2] - exact[:, :2], axis=1))
            error_e = np.max(np.abs(energy(states)[2] - exact_energy))
            print(f"{name:<9} {h:7.4f}          {error_r:12.5e}          {error_e:12.5e}")


def compare_resistance():
    # Точные траектории показывают влияние tau без ошибки численного метода.
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    print("\nВлияние сопротивления (точное решение):")
    print("tau, с           Время полёта, с     Дальность, м     Высота, м")
    for tau in TAU_VALUES:
        t_end = flight_time(tau)
        t = np.linspace(0, t_end, 1001)
        states = exact_solution(t, tau)
        peak_t = STATE_0[3] / G if np.isinf(tau) else tau * np.log1p(STATE_0[3] / (G * tau))
        height = exact_solution(peak_t, tau)[0, 1]
        distance = states[-1, 0] - STATE_0[0]
        label = "Без сопротивления" if np.isinf(tau) else f"tau = {tau:g} с"
        ax.plot(states[:, 0], states[:, 1], label=label)
        tau_text = "inf" if np.isinf(tau) else f"{tau:g}"
        print(f"{tau_text:<12}     {t_end:10.4f}         {distance:10.4f}     {height:10.4f}")
    ax.set(title="Влияние сопротивления: одинаковая начальная скорость и угол",
           xlabel="x, м", ylabel="y, м")
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.legend()


if __name__ == "__main__":
    results = calculate()
    plot_results(results)
    compare_errors(results)
    compare_resistance()
    plt.show()
