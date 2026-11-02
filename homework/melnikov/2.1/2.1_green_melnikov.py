import numpy as np
import matplotlib.pyplot as plt


# Два точечных груза на невесомых жёстких стержнях, трения нет.
g = 9.81
m1 = 1.0
m2 = 1.0
L1 = 1.0
L2 = 1.0
T = 20.0
steps = [0.04, 0.02, 0.01]

# Оба угла отсчитываются от вертикали вниз, скорости — угловые.
theta1_0 = np.deg2rad(5.0)
theta2_0 = np.deg2rad(10.0)
v1_0 = 0.0
v2_0 = 0.0
s0 = np.array([theta1_0, theta2_0, v1_0, v2_0])


def f(s):
    theta1, theta2, v1, v2 = s
    delta = theta1 - theta2

    # Связанные уравнения: M11*a1 + M12*a2 = b1, M12*a1 + M22*a2 = b2.
    M11 = (m1 + m2) * L1**2
    M12 = m2 * L1 * L2 * np.cos(delta)
    M22 = m2 * L2**2
    b1 = -m2 * L1 * L2 * v2**2 * np.sin(delta) - (m1 + m2) * g * L1 * np.sin(theta1)
    b2 = m2 * L1 * L2 * v1**2 * np.sin(delta) - m2 * g * L2 * np.sin(theta2)

    det = M11 * M22 - M12**2
    a1 = (M22 * b1 - M12 * b2) / det
    a2 = (M11 * b2 - M12 * b1) / det
    return np.array([v1, v2, a1, a2])


def rk4_step(s, dt):
    k1 = f(s)
    k2 = f(s + dt * k1 / 2)
    k3 = f(s + dt * k2 / 2)
    k4 = f(s + dt * k3)
    return s + dt * (k1 + 2*k2 + 2*k3 + k4) / 6


def integrate_rk4(dt):
    n = int(round(T / dt))
    t = np.arange(n + 1) * dt
    traj = np.zeros((n + 1, 4))
    traj[0] = s0

    for i in range(n):
        traj[i + 1] = rk4_step(traj[i], dt)

    return t, traj


def integrate_ab2(dt):
    n = int(round(T / dt))
    t = np.arange(n + 1) * dt
    traj = np.zeros((n + 1, 4))
    traj[0] = s0
    traj[1] = rk4_step(s0, dt)  # Первый шаг для запуска многошагового метода.

    for i in range(1, n):
        traj[i + 1] = traj[i] + dt * (3*f(traj[i]) - f(traj[i - 1])) / 2

    return t, traj


def lower_bob(traj):
    theta1 = traj[:, 0]
    theta2 = traj[:, 1]
    x2 = L1 * np.sin(theta1) + L2 * np.sin(theta2)
    y2 = -L1 * np.cos(theta1) - L2 * np.cos(theta2)
    return x2, y2


def total_energy(traj):
    theta1 = traj[:, 0]
    theta2 = traj[:, 1]
    v1 = traj[:, 2]
    v2 = traj[:, 3]

    kinetic = (
        0.5 * (m1 + m2) * L1**2 * v1**2
        + 0.5 * m2 * L2**2 * v2**2
        + m2 * L1 * L2 * v1 * v2 * np.cos(theta1 - theta2)
    )
    # Ноль потенциальной энергии — в нижнем положении равновесия.
    potential = (
        (m1 + m2) * g * L1 * (1 - np.cos(theta1))
        + m2 * g * L2 * (1 - np.cos(theta2))
    )
    return kinetic + potential


# Углы и траектории: отдельный столбец для каждого шага.
fig1, ax1 = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey="row")
fig1.suptitle("Углы двойного маятника относительно вертикали вниз")
angle_names = ["Угол верхнего стержня", "Угол нижнего стержня"]
fig2, ax2 = plt.subplots(1, 3, figsize=(13, 4))
fig2.suptitle(f"Траектория нижнего груза за всё время расчёта: 0–{T:g} с")
fig3, ax3 = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig3.suptitle("Полная механическая энергия")

E0 = total_energy(s0.reshape(1, 4))[0]
print(f"Начальная энергия: {E0:.8f} Дж")
print("Метод     Шаг, с    Максимальное изменение энергии, %")

for j, dt in enumerate(steps):
    t_rk, traj_rk = integrate_rk4(dt)
    t_ab, traj_ab = integrate_ab2(dt)
    E_rk = total_energy(traj_rk)
    E_ab = total_energy(traj_ab)

    for k in range(2):
        ax1[k, j].plot(t_rk, np.rad2deg(traj_rk[:, k]), label="RK4")
        ax1[k, j].plot(t_ab, np.rad2deg(traj_ab[:, k]), "--", label="Адамс–Бэшфорт 2")
        ax1[k, j].set_title(f"{angle_names[k]}\nШаг интегрирования {dt:g} с")
        ax1[k, j].grid(True)
        ax1[k, j].legend(fontsize=8)
    ax1[1, j].set_xlabel("Время, с")

    # Используем все положения нижнего груза от начала до конца расчёта.
    x_rk, y_rk = lower_bob(traj_rk)
    x_ab, y_ab = lower_bob(traj_ab)
    ax2[j].plot(x_rk, y_rk, label="RK4")
    ax2[j].plot(x_ab, y_ab, "--", label="Адамс–Бэшфорт 2")
    ax2[j].plot(x_rk[0], y_rk[0], "ko", markersize=4, label="Начало")
    ax2[j].set_title(f"Шаг интегрирования {dt:g} с")
    ax2[j].set_xlabel("x₂, м")
    ax2[j].set_ylabel("y₂, м")
    ax2[j].set_aspect("equal", adjustable="datalim")
    ax2[j].grid(True)
    ax2[j].legend(fontsize=8)

    ax3[0].plot(t_rk, E_rk, label=f"Шаг {dt:g} с")
    ax3[1].plot(t_ab, E_ab, label=f"Шаг {dt:g} с")

    # Изменение относительно E(0): 100 * max|E(t) - E(0)| / E(0).
    drift_rk = 100 * np.max(np.abs(E_rk - E0)) / E0
    drift_ab = 100 * np.max(np.abs(E_ab - E0)) / E0
    print(f"RK4       {dt:6.3f}             {drift_rk:.6f}")
    print(f"AB2       {dt:6.3f}             {drift_ab:.6f}")

ax1[0, 0].set_ylabel("Угол, градусы")
ax1[1, 0].set_ylabel("Угол, градусы")

for k, method in enumerate(["RK4", "Адамс–Бэшфорт 2"]):
    ax3[k].axhline(E0, color="black", linestyle=":", label="Начальная энергия")
    ax3[k].set_title(method)
    ax3[k].set_ylabel("Энергия, Дж")
    ax3[k].ticklabel_format(axis="y", style="plain", useOffset=False)
    ax3[k].grid(True)
    ax3[k].legend()
ax3[1].set_xlabel("Время, с")

fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
plt.show()
