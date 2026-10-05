import numpy as np
import matplotlib.pyplot as plt

g = 9.81
l1, l2 = 1.0, 1.0
m1, m2 = 1.0, 1.0

theta1_0 = 0.15
theta2_0 = -0.10
omega1_0 = 0.0
omega2_0 = 0.0

y0 = np.array([theta1_0, theta2_0, omega1_0, omega2_0])
t_max = 15.0
step_sizes = [0.01, 0.03, 0.06]

def derivatives(t, y):
    th1, th2, w1, w2 = y
    delta = th1 - th2

    a11 = (m1 + m2) * l1
    a12 = m2 * l2 * np.cos(delta)
    b1 = -m2 * l2 * (w2**2) * np.sin(delta) - (m1 + m2) * g * np.sin(th1)

    a21 = l1 * np.cos(delta)
    a22 = l2
    b2 = l1 * (w1**2) * np.sin(delta) - g * np.sin(th2)

    det = a11 * a22 - a12 * a21
    eps1 = (b1 * a22 - b2 * a12) / det
    eps2 = (a11 * b2 - a21 * b1) / det

    return np.array([w1, w2, eps1, eps2])

def rk4_step(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(t + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def solve_rk4(y0, t_max, h):
    t_vals = np.arange(0, t_max + h, h)
    n = len(t_vals)
    y_vals = np.zeros((n, 4))
    y_vals[0] = y0

    for i in range(n - 1):
        y_vals[i + 1] = rk4_step(derivatives, t_vals[i], y_vals[i], h)

    return t_vals, y_vals

def solve_ab2(y0, t_max, h):
    t_vals = np.arange(0, t_max + h, h)
    n = len(t_vals)
    y_vals = np.zeros((n, 4))
    y_vals[0] = y0

    y_vals[1] = rk4_step(derivatives, t_vals[0], y_vals[0], h)

    f_prev = derivatives(t_vals[0], y_vals[0])
    f_curr = derivatives(t_vals[1], y_vals[1])

    for i in range(1, n - 1):
        y_vals[i + 1] = y_vals[i] + h * (1.5 * f_curr - 0.5 * f_prev)
        f_prev = f_curr
        f_curr = derivatives(t_vals[i + 1], y_vals[i + 1])

    return t_vals, y_vals

def compute_energy(y_vals):
    th1 = y_vals[:, 0]
    th2 = y_vals[:, 1]
    w1 = y_vals[:, 2]
    w2 = y_vals[:, 3]

    T = 0.5 * (m1 + m2) * (l1**2) * (w1**2) + \
        0.5 * m2 * (l2**2) * (w2**2) + \
        m2 * l1 * l2 * w1 * w2 * np.cos(th1 - th2)

    V = -(m1 + m2) * g * l1 * np.cos(th1) - m2 * g * l2 * np.cos(th2)
    return T + V

def get_coords(y_vals):
    th1 = y_vals[:, 0]
    th2 = y_vals[:, 1]
    x1 = l1 * np.sin(th1)
    y1 = -l1 * np.cos(th1)
    x2 = x1 + l2 * np.sin(th2)
    y2 = y1 - l2 * np.cos(th2)
    return x1, y1, x2, y2

results_rk4 = {}
results_ab2 = {}

for h in step_sizes:
    t_rk, y_rk = solve_rk4(y0, t_max, h)
    t_ab, y_ab = solve_ab2(y0, t_max, h)
    E_rk = compute_energy(y_rk)
    E_ab = compute_energy(y_ab)
    results_rk4[h] = {"t": t_rk, "y": y_rk, "E": E_rk}
    results_ab2[h] = {"t": t_ab, "y": y_ab, "E": E_ab}

    err_rk = np.max(np.abs(E_rk - E_rk[0]))
    err_ab = np.max(np.abs(E_ab - E_ab[0]))
    print(f"h = {h:.2f} | max |dE| RK4: {err_rk:.2e} | max |dE| AB2: {err_ab:.2e}")

h_demo = 0.01
t_d = results_rk4[h_demo]["t"]
y_rk_d = results_rk4[h_demo]["y"]
y_ab_d = results_ab2[h_demo]["y"]

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t_d, y_rk_d[:, 0], label="θ₁ (RK4)", color="blue")
plt.plot(t_d, y_ab_d[:, 0], "--", label="θ₁ (AB2)", color="cyan")
plt.title(f"Колебания углов двойного маятника (h = {h_demo} c)")
plt.ylabel("θ₁, рад")
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t_d, y_rk_d[:, 1], label="θ₂ (RK4)", color="red")
plt.plot(t_d, y_ab_d[:, 1], "--", label="θ₂ (AB2)", color="orange")
plt.xlabel("Время t, с")
plt.ylabel("θ₂, рад")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(8, 6))
colors = ["green", "blue", "magenta"]

for i, h in enumerate(step_sizes):
    _, _, x2_rk, y2_rk = get_coords(results_rk4[h]["y"])
    _, _, x2_ab, y2_ab = get_coords(results_ab2[h]["y"])
    plt.plot(x2_rk, y2_rk, label=f"RK4 (h={h})", color=colors[i])
    plt.plot(x2_ab, y2_ab, ":", label=f"AB2 (h={h})", color=colors[i])

_, _, x2_0, y2_0 = get_coords(np.array([y0]))
plt.scatter(x2_0, y2_0, color="red", zorder=5, label="Старт")
plt.title("Траектория нижнего груза (x₂, y₂)")
plt.xlabel("x₂, м")
plt.ylabel("y₂, м")
plt.grid(True)
plt.legend()
plt.tight_layout()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

for h in step_sizes:
    ax1.plot(results_rk4[h]["t"], results_rk4[h]["E"], label=f"RK4, h = {h}")
    ax2.plot(results_ab2[h]["t"], results_ab2[h]["E"], label=f"AB2, h = {h}")

ax1.set_title("Полная механическая энергия (RK4)")
ax1.set_ylabel("E, Дж")
ax1.ticklabel_format(useOffset=False)
ax1.grid(True)
ax1.legend()

ax2.set_title("Полная механическая энергия (AB2)")
ax2.set_xlabel("Время t, с")
ax2.set_ylabel("E, Дж")
ax2.grid(True)
ax2.legend()
plt.tight_layout()

plt.show()
