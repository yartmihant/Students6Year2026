import numpy as np

import matplotlib.pyplot as plt


OMEGA = 1.0
THETA_0 = 0.1
V_0 = 0.0
T_END = 20.0
STEPS = (0.1, 0.05, 0.025, 0.0125)


def rhs(state, omega):
    theta, v = state
    return np.array([v, -omega**2 * theta])


def euler_step(state, h, omega):
    return state + h * rhs(state, omega)


def rk4_step(state, h, omega):
    k1 = rhs(state, omega)
    k2 = rhs(state + h * k1 / 2, omega)
    k3 = rhs(state + h * k2 / 2, omega)
    k4 = rhs(state + h * k3, omega)
    return state + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def integrate(step, h, t_end, state_0, omega):

    n = int(np.ceil(t_end / h))
    t = np.minimum(np.arange(n + 1, dtype=float) * h, t_end)
    t[-1] = t_end
    states = np.empty((len(t), 2), dtype=float)
    states[0] = state_0
    for i in range(len(t) - 1):
        states[i + 1] = step(states[i], t[i + 1] - t[i], omega)
    return t, states


def exact_solution(t, theta_0, v_0, omega):
    theta = theta_0 * np.cos(omega * t) + v_0 / omega * np.sin(omega * t)
    v = -theta_0 * omega * np.sin(omega * t) + v_0 * np.cos(omega * t)
    return np.column_stack((theta, v))


def energy(states, omega):
    potential = omega**2 * states[:, 0]**2 / 2
    kinetic = states[:, 1]**2 / 2
    return potential, kinetic, potential + kinetic


def calculate():
    results = []
    for h in STEPS:
        trajectories = {}
        for name, step in (("Эйлер", euler_step), ("RK4", rk4_step)):
            t, states = integrate(step, h, T_END, [THETA_0, V_0], OMEGA)
            trajectories[name] = states
        exact = exact_solution(t, THETA_0, V_0, OMEGA)
        results.append((h, t, exact, trajectories))
    return results


def plot_results(results):
    colors = {"Эйлер": "#d55e00", "RK4": "#0072b2"}
    fine_t = np.linspace(0, T_END, 4001)
    fine_exact = exact_solution(fine_t, THETA_0, V_0, OMEGA)
    ncols = min(2, len(results))
    nrows = (len(results) + ncols - 1) // ncols

    for phase, title in (
        (False, "Маятник: сравнение угла с точным решением"),
        (True, "Фазовые траектории маятника"),
    ):
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows),
                                 squeeze=False, layout="constrained")
        fig.suptitle(title)
        for ax, (h, t, exact, trajectories) in zip(axes.flat, results):
            for name, states in trajectories.items():
                x = states[:, 0] if phase else t
                y = states[:, 1] if phase else states[:, 0]
                ax.plot(x, y, color=colors[name], label=name, linewidth=1.4)
            ax.plot(fine_exact[:, 0] if phase else fine_t,
                    fine_exact[:, 1] if phase else fine_exact[:, 0],
                    "k--", label="Точное решение", linewidth=1.3)
            ax.set(title=f"h = {h:g} с", xlabel="θ, рад" if phase else "t, с",
                   ylabel="dθ/dt, рад/с" if phase else "θ, рад")
            if phase:
                ax.scatter([THETA_0], [V_0], s=25, color="black", zorder=5)
                ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=9)
        for ax in list(axes.flat)[len(results):]:
            ax.set_visible(False)

    fig, axes = plt.subplots(len(results), 3, figsize=(15, 3 * len(results)),
                             squeeze=False, layout="constrained")
    fig.suptitle("Энергии линейной модели")
    fine_energies = energy(fine_exact, OMEGA)
    for row, (h, t, exact, trajectories) in enumerate(results):
        for col, name_energy in enumerate(("Потенциальная U", "Кинетическая K", "Полная E")):
            ax = axes[row, col]
            for name, states in trajectories.items():
                ax.plot(t, energy(states, OMEGA)[col], color=colors[name], label=name)
            ax.plot(fine_t, fine_energies[col], "k--", label="Точное решение", linewidth=1.2)
            ax.set(title=f"{name_energy}, h = {h:g} с", xlabel="t, с",
                   ylabel="Энергия / (mℓ²), с⁻²")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)


def compare_errors(results):
    rows = []
    print("\nМетод       h       max|theta error|  max|v error|     max|dE|/E0")
    for h, t, exact, trajectories in results:
        e0 = energy(exact, OMEGA)[2][0]
        for name, states in trajectories.items():
            err_theta, err_v = np.max(np.abs(states - exact), axis=0)
            drift = np.max(np.abs(energy(states, OMEGA)[2] - e0))
            relative_drift = drift / e0 if e0 > 0 else 0.0
            rows.append((name, h, err_theta, err_v, relative_drift))
            print(f"{name:<9} {h:7.4f}    {err_theta:12.5e}    {err_v:12.5e}    {relative_drift:12.5e}")

    print("\nНаблюдаемый порядок по максимальной ошибке угла:")
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for name, color in (("Эйлер", "#d55e00"), ("RK4", "#0072b2")):
        selected = sorted((row for row in rows if row[0] == name), key=lambda x: -x[1])
        hs = np.array([row[1] for row in selected])
        errors = np.array([row[2] for row in selected])
        ax.loglog(hs, np.maximum(errors, np.finfo(float).tiny), "o-", color=color, label=name)
        for i in range(len(hs) - 1):
            if errors[i] > 0 and errors[i + 1] > 0 and hs[i] != hs[i + 1]:
                p = np.log(errors[i] / errors[i + 1]) / np.log(hs[i] / hs[i + 1])
                print(f"{name}: h={hs[i]:g} -> {hs[i + 1]:g}: p={p:.4f}")
    ax.set(title="Ошибка угла при уменьшении шага", xlabel="h, с",
           ylabel="max |θ числ. − θ точн.|, рад")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()


if __name__ == "__main__":
    results = calculate()
    plot_results(results)
    compare_errors(results)
    plt.show()
