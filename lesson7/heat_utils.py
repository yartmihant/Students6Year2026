"""Учебные одномерные схемы теплопроводности. Все величины безразмерные."""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve_banded


@dataclass
class Operator:
    """L(U,t) = A U + b(t); только внутренние неизвестные."""
    x: np.ndarray
    lower: np.ndarray
    diag: np.ndarray
    upper: np.ndarray
    boundary: object
    flux: object = None
    h: float = 0.0

    def apply(self, u):
        out = self.diag * u
        out[1:] += self.lower * u[:-1]
        out[:-1] += self.upper * u[1:]
        return out


def nodal_operator(n=100, alpha=1.0, left=lambda t: 0., right=lambda t: 0.):
    """N интервалов, неизвестные в N-1 внутренних узлах; Дирихле."""
    if n < 3 or alpha <= 0:
        raise ValueError('Нужны n >= 3 и alpha > 0')
    h = 1. / n
    x = np.arange(1, n) * h
    a = alpha / h**2

    def boundary(t):
        b = np.zeros(n - 1)
        b[0] = a * left(t)
        b[-1] = a * right(t)
        return b

    return Operator(x, np.full(n-2, a), np.full(n-1, -2*a),
                    np.full(n-2, a), boundary, h=h)


def composite_operator(n=100, k_left=1., k_right=.1, contact=0.,
                       left=lambda t: 1., right=lambda t: 0.):
    """N ячеек; стык на грани x=1/2; сопротивления между центрами."""
    if n < 4 or n % 2 or min(k_left, k_right) <= 0 or contact < 0:
        raise ValueError('Нужно чётное n >= 4, положительные k и contact >= 0')
    h = 1. / n
    x = (np.arange(n) + .5) * h
    k = np.where(x < .5, k_left, k_right)
    resistances = np.empty(n + 1)
    resistances[0] = h / (2*k[0])
    resistances[-1] = h / (2*k[-1])
    resistances[1:-1] = h/(2*k[:-1]) + h/(2*k[1:])
    resistances[n//2] += contact
    conductance = 1 / resistances

    def boundary(t):
        b = np.zeros(n)
        b[0] = conductance[0] * left(t) / h
        b[-1] = conductance[-1] * right(t) / h
        return b

    def flux(u, t):
        # Все потоки положительны вправо, включая граничные.
        return -conductance * np.diff(np.r_[left(t), u, right(t)])

    return Operator(x, conductance[1:-1]/h,
                    -(conductance[:-1]+conductance[1:])/h,
                    conductance[1:-1]/h, boundary, flux, h)


def theta_step(op, u, t, dt, theta):
    """Один шаг; известные граничные значения учтены на двух слоях."""
    rhs = u + (1-theta)*dt*op.apply(u)
    rhs += dt*((1-theta)*op.boundary(t) + theta*op.boundary(t+dt))
    if theta == 0:
        return rhs
    ab = np.zeros((3, len(u)))
    ab[0, 1:] = -theta*dt*op.upper
    ab[1] = 1-theta*dt*op.diag
    ab[2, :-1] = -theta*dt*op.lower
    return solve_banded((1, 1), ab, rhs, check_finite=False)


def evolve(op, u0, times, dt, theta=.5, allow_unstable=False):
    """Дойти точно до заданных времён; баланс с квадратурой theta-схемы.

    times начинается с нуля. dt — верхний предел шага. До каждой
    точки вывода последний шаг сокращается. Храним только запрошенные слои.
    """
    times = np.asarray(times, dtype=float)
    if len(times) < 2 or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError('times: 0 и строго возрастающие положительные времена')
    if dt <= 0 or theta not in (0., .5, 1.):
        raise ValueError('Нужны dt > 0 и theta из {0, 0.5, 1}')
    if theta == 0 and dt*np.max(-op.diag) > 1+1e-12 and not allow_unstable:
        raise ValueError('Нарушено достаточное условие монотонности явной схемы')
    u = np.asarray(u0, dtype=float).copy()
    if u.shape != op.x.shape:
        raise ValueError('Начальный профиль не соответствует сетке')
    history = [u.copy()]
    incoming, outgoing = [0.], [0.]
    total_in = total_out = 0.
    t = 0.
    steps = 0
    for target in times[1:]:
        while target-t > 1e-14 * max(1., target):
            step = min(dt, target-t)
            new = theta_step(op, u, t, step, theta)
            if not np.all(np.isfinite(new)) or np.max(np.abs(new)) > 1e12:
                raise FloatingPointError(f'Рост решения: расчёт остановлен при t={t+step:g}')
            if op.flux is not None:
                old_q, new_q = op.flux(u, t), op.flux(new, t+step)
                total_in += step*((1-theta)*old_q[0] + theta*new_q[0])
                total_out += step*((1-theta)*old_q[-1] + theta*new_q[-1])
            u, t = new, t+step
            steps += 1
        t = float(target)
        history.append(u.copy())
        incoming.append(total_in)
        outgoing.append(total_out)
    result = dict(t=times, u=np.array(history), steps=steps,
                  incoming=np.array(incoming), outgoing=np.array(outgoing))
    if op.flux is not None:
        heat = op.h * result['u'].sum(axis=1)
        result['heat'] = heat
        result['balance'] = heat-heat[0]-result['incoming']+result['outgoing']
    return result


def cooling_exact(x, t):
    return np.exp(-np.pi**2*t)*np.sin(np.pi*x) + .3*np.exp(-25*np.pi**2*t)*np.sin(5*np.pi*x)


def stationary_composite(x, k_left=1., k_right=.1, contact=0.):
    q = 1/(.5/k_left + contact + .5/k_right)
    resistance = np.where(x < .5, x/k_left,
                          .5/k_left + contact + (x-.5)/k_right)
    return 1-q*resistance, q


def periodic_amplitude(x, omega=8*np.pi, alpha=1.):
    lam = (1+1j)*np.sqrt(omega/(2*alpha))
    return np.sinh(lam*(1-np.asarray(x))) / np.sinh(lam)
