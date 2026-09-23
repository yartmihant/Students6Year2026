import numpy as np
import matplotlib.pyplot as plt

# ПАРАМЕТРЫ
omega   = 1.0     # частота
theta_0 = 1.0     # начальный угол
v_0     = 0.0     # начальная скорость
T       = 100.0   # время интегрирования

# ФИЗИКА
def f(s):
    theta, v = s
    return np.array([v, -omega**2 * theta])

def analytic(t):
    theta = theta_0 * np.cos(omega * t) + (v_0 / omega) * np.sin(omega * t)
    v     = -theta_0 * omega * np.sin(omega * t) + v_0 * np.cos(omega * t)
    return theta, v

# ЧИСЛЕННЫЕ МЕТОДЫ
def euler_explicit_step(s, dt):
    return s + dt * f(s)

def euler_implicit_step(s, dt):
    theta, v = s
    det = 1.0 + dt**2 * omega**2
    theta_new = (theta + dt * v) / det
    v_new     = (v - dt * omega**2 * theta) / det
    return np.array([theta_new, v_new])

def rk4_step(s, dt):
    k1 = f(s)
    k2 = f(s + dt * k1 / 2)
    k3 = f(s + dt * k2 / 2)
    k4 = f(s + dt * k3)
    return s + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

def integrate(step, dt, T):
    n = int(round(T / dt))
    t = np.linspace(0, T, n + 1)
    traj = np.zeros((n + 1, 2))
    traj[0] = [theta_0, v_0]
    for i in range(1, n + 1):
        traj[i] = step(traj[i - 1], dt)
    return t, traj

# ЭНЕРГИИ
def energies(traj):
    theta = traj[:, 0]
    v     = traj[:, 1]
    Ek = 0.5 * v**2
    Ep = 0.5 * omega**2 * theta**2
    return Ek, Ep

# РАСЧЁТЫ
dt    = 0.1     # для theta(t) и энергий
dt_ph = 0.1     # для фазовой траектории

t_an = np.linspace(0, T, 5000)
theta_an, v_an = analytic(t_an)

# при dt = 0.1 (для theta(t) и энергий)
t_e, traj_e = integrate(euler_explicit_step, dt, T)
t_i, traj_i = integrate(euler_implicit_step, dt, T)
t_r, traj_r = integrate(rk4_step,            dt, T)

# при dt = 0.1 (для фазовой траектории)
t_ph_an = np.linspace(0, T, 5000)
theta_ph_an, v_ph_an = analytic(t_ph_an)

t_e_ph, traj_e_ph = integrate(euler_explicit_step, dt_ph, T)
t_i_ph, traj_i_ph = integrate(euler_implicit_step, dt_ph, T)
t_r_ph, traj_r_ph = integrate(rk4_step,            dt_ph, T)

# ==================== ОКНО 1: theta(t) ====================
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(t_an, theta_an, 'k', linewidth=2, label='Аналитическое')
ax1.plot(t_e, traj_e[:, 0], '--', label='Эйлер явный')
ax1.plot(t_i, traj_i[:, 0], ':',  label='Эйлер неявный')
ax1.plot(t_r, traj_r[:, 0], '-',  label='RK4')
ax1.set_xlabel('t'); ax1.set_ylabel('theta')
ax1.set_title(f'theta(t): сравнение с аналитическим решением (dt = {dt})')
ax1.legend(); ax1.grid(True)

# ==================== ОКНО 2: фазовая траектория ====================
fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.plot(theta_ph_an, v_ph_an, 'k', linewidth=2, label='Аналитическое')
ax2.plot(traj_e_ph[:, 0], traj_e_ph[:, 1], '--', label='Эйлер явный')
ax2.plot(traj_i_ph[:, 0], traj_i_ph[:, 1], ':',  label='Эйлер неявный')
ax2.plot(traj_r_ph[:, 0], traj_r_ph[:, 1], '-',  label='RK4')
ax2.set_xlabel('theta'); ax2.set_ylabel('v')
ax2.set_title(f'Фазовая траектория (dt = {dt_ph})')
ax2.legend(); ax2.grid(True); ax2.set_aspect('equal')

# ==================== ОКНО 3: энергии ====================
fig3, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

# --- RK4 ---
Ek, Ep = energies(traj_r)
axs[0].plot(t_r, Ek,      label='Кинетическая')
axs[0].plot(t_r, Ep,      label='Потенциальная')
axs[0].plot(t_r, Ek + Ep, label='Полная', linestyle='--')
axs[0].set_ylabel('E'); axs[0].set_title(f'Энергии, RK4 (dt = {dt})')
axs[0].legend(); axs[0].grid(True)

# --- Явный Эйлер ---
Ek, Ep = energies(traj_e)
axs[1].plot(t_e, Ek,      label='Кинетическая')
axs[1].plot(t_e, Ep,      label='Потенциальная')
axs[1].plot(t_e, Ek + Ep, label='Полная', linestyle='--')
axs[1].set_ylabel('E'); axs[1].set_title(f'Энергии, Эйлер явный (dt = {dt})')
axs[1].legend(); axs[1].grid(True)

# --- Неявный Эйлер ---
Ek, Ep = energies(traj_i)
axs[2].plot(t_i, Ek,      label='Кинетическая')
axs[2].plot(t_i, Ep,      label='Потенциальная')
axs[2].plot(t_i, Ek + Ep, label='Полная', linestyle='--')
axs[2].set_xlabel('t'); axs[2].set_ylabel('E')
axs[2].set_title(f'Энергии, Эйлер неявный (dt = {dt})')
axs[2].legend(); axs[2].grid(True)

fig3.tight_layout()

plt.show()