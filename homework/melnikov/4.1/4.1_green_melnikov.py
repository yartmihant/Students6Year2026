"""Метод стрельбы для возвращения маятника в заданное положение.

Уравнение: theta'' + theta = 0, theta(0) = 0, theta(2) = 0.2.
Неизвестная начальная скорость s = theta'(0).

Зависимости: numpy, matplotlib (pip install numpy matplotlib).
Запуск: python 4.1_green_melnikov.py
Без окон: python 4.1_green_melnikov.py --no-gui
Сохранить графики: python 4.1_green_melnikov.py --save-plots graphs

Интегрирование — собственная реализация Рунге — Кутты 4-го порядка.
Производная в методе Ньютона вычисляется центральной разностью.
Все три окна создаются до единственного вызова plt.show().
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import math
import warnings

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("Установите зависимости: python -m pip install numpy matplotlib") from exc


T = 2.0
TARGET = 0.2
S_EXACT = TARGET / math.sin(T)
STEPS = (0.4, 0.2, 0.1, 0.05, 0.025)
ROOT_TOL = 1e-10                 # Абсолютный допуск по |R(s)|.
BRACKET = (0.0, 0.5)            # На концах невязка имеет разные знаки.
NEWTON_START = 0.5


def exact_solution(t):
    return S_EXACT * np.sin(t)


def integrate(s: float, h: float):
    """Решить задачу Коши theta'=v, v'=-theta методом RK4.

    При необходимости шаг немного уменьшается, чтобы последний узел был t=2.
    Точное решение используется только вне этой функции, для сравнения.
    """
    if not math.isfinite(h) or h <= 0:
        raise ValueError("Шаг должен быть положительным конечным числом.")
    n = max(1, math.ceil(T / h - 1e-12))
    dt = T / n
    t = np.linspace(0.0, T, n + 1)
    theta = np.empty(n + 1)
    velocity = np.empty(n + 1)
    theta[0], velocity[0] = 0.0, s
    for i in range(n):
        y, v = theta[i], velocity[i]
        k1y, k1v = v, -y
        k2y, k2v = v + dt * k1v / 2, -(y + dt * k1y / 2)
        k3y, k3v = v + dt * k2v / 2, -(y + dt * k2y / 2)
        k4y, k4v = v + dt * k3v, -(y + dt * k3y)
        theta[i + 1] = y + dt * (k1y + 2 * k2y + 2 * k3y + k4y) / 6
        velocity[i + 1] = v + dt * (k1v + 2 * k2v + 2 * k3v + k4v) / 6
    return t, theta, velocity


@dataclass
class Residual:
    h: float
    calls: int = 0

    def __call__(self, s: float) -> float:
        # Каждый вызов означает отдельное интегрирование, кэширования нет.
        self.calls += 1
        return float(integrate(s, self.h)[1][-1] - TARGET)


@dataclass
class RootResult:
    method: str
    s: float
    residual: float
    iterations: int
    calls: int
    history: list[tuple[float, float]] = field(default_factory=list)


def newton(h: float, tol: float, start: float = NEWTON_START) -> RootResult:
    """Ньютон с численной производной; учтены оба дополнительных вызова R."""
    residual = Residual(h)
    s = start
    history = []
    for iteration in range(31):
        r = residual(s)
        history.append((s, r))
        if abs(r) <= tol:
            return RootResult("Ньютон", s, r, iteration, residual.calls, history)
        if iteration == 30:
            break
        delta = np.cbrt(np.finfo(float).eps) * max(1.0, abs(s))
        derivative = (residual(s + delta) - residual(s - delta)) / (2 * delta)
        if not math.isfinite(derivative) or abs(derivative) < 1e-14:
            raise RuntimeError("Ньютон: численная производная слишком мала.")
        s -= r / derivative
        if not math.isfinite(s):
            raise RuntimeError("Ньютон: получено неконечное приближение.")
    raise RuntimeError("Ньютон не достиг заданного допуска за 30 итераций.")


def bisection(h: float, tol: float, bracket=BRACKET) -> RootResult:
    """Бисекция со строгой проверкой смены знака и остановкой по |R|."""
    residual = Residual(h)
    a, b = bracket
    if a >= b:
        raise ValueError("Требуется a < b.")
    ra, rb = residual(a), residual(b)
    for s, r in ((a, ra), (b, rb)):
        if abs(r) <= tol:
            return RootResult("Бисекция", s, r, 0, residual.calls, [(s, r)])
    if ra * rb >= 0:
        raise ValueError("Для бисекции нужна смена знака R на концах интервала.")
    history = []
    for iteration in range(1, 201):
        s = (a + b) / 2
        r = residual(s)
        history.append((s, r))
        if abs(r) <= tol:
            return RootResult("Бисекция", s, r, iteration, residual.calls, history)
        if s == a or s == b:
            raise RuntimeError("Бисекция достигла предела машинной точности.")
        if ra * r < 0:
            b, rb = s, r
        else:
            a, ra = s, r
    raise RuntimeError("Бисекция не достигла заданного допуска за 200 итераций.")


def trajectory_error(result: RootResult, h: float) -> float:
    t, y, _ = integrate(result.s, h)
    return float(np.max(np.abs(y - exact_solution(t))))


def compute_experiments():
    # Для КАЖДОГО шага и допуска оба метода запускаются заново.
    runs = []
    for h in STEPS:
        nr, br = newton(h, ROOT_TOL), bisection(h, ROOT_TOL)
        runs.append((h, nr, br))
    tolerances = np.array([1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14])
    study = {}
    for h in (0.2, 0.025):
        for method in (newton, bisection):
            results = [method(h, float(tol)) for tol in tolerances]
            errors = np.array([trajectory_error(result, h) for result in results])
            study[h, method.__name__] = (results, errors)
    return runs, tolerances, study


def print_report(runs, tolerances, study):
    print(f"s* = {S_EXACT:.14f}    допуск = {ROOT_TOL:.0e}    RK4")
    print(f"Бисекция: {BRACKET}    Ньютон: s0 = {NEWTON_START}")
    print()
    print(f"{'h':>6} {'Метод':>9} {'s':>17} {'|R_h|':>11} "
          f"{'Emax':>11} {'|s-s*|':>11} {'Итер.':>6} {'Выз. R':>7}")
    for h, nr, br in runs:
        for result in (nr, br):
            print(f"{h:6.3f} {result.method:>9} {result.s:17.12f} "
                  f"{abs(result.residual):11.3e} {trajectory_error(result, h):11.3e} "
                  f"{abs(result.s-S_EXACT):11.3e} {result.iterations:6d} {result.calls:7d}")
    print("Emax — максимальная ошибка в узлах; вызовы R включают численную производную.")

    print("\nДопуск и Emax (бисекция):")
    print(f"{'Допуск':>10} {'h = 0.2':>12} {'Выз. R':>7} "
          f"{'h = 0.025':>12} {'Выз. R':>7}")
    coarse_results, coarse_errors = study[0.2, 'bisection']
    fine_results, fine_errors = study[0.025, 'bisection']
    for i in range(0, len(tolerances), 2):
        print(f"{tolerances[i]:10.0e} {coarse_errors[i]:12.3e} {coarse_results[i].calls:7d} "
              f"{fine_errors[i]:12.3e} {fine_results[i].calls:7d}")

    print("\nОтвет: меньший допуск не устраняет ошибку интегрирования.")
    print("Чтобы повысить точность, нужно уменьшить шаг h и заново подобрать s.")


def make_figures(plt, runs):
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.titlesize': 10, 'axes.labelsize': 9,
        'legend.fontsize': 8, 'axes.grid': True, 'grid.alpha': 0.2,
        'lines.linewidth': 1.7,
    })
    figures = []
    dense_t = np.linspace(0, T, 501)
    h = 0.2
    _, nr, br = next(row for row in runs if row[0] == h)

    def title(fig, text):
        # В заголовках окон — формулировки артефактов без сокращений.
        fig.canvas.manager.set_window_title(text.replace('\n', ' '))
        fig.suptitle(text, fontsize=11)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), layout='constrained')
    figures.append(fig)
    title(fig, 'Построить несколько пристрелочных траекторий и отметить\n'
               'требуемое положение в конечный момент')
    for s in (0.10, 0.16, 0.28, 0.36):
        t, y, _ = integrate(s, h)
        line, = ax.plot(t, y, label=f's = {s:.2f}', alpha=0.75)
        ax.plot(T, y[-1], 'o', color=line.get_color(), markersize=4)
    t, y, _ = integrate(nr.s, h)
    ax.plot(t, y, color='#146647', linewidth=2.7, label=f'Подобранная s = {nr.s:.7f}')
    ax.scatter([T], [TARGET], s=150, marker='*', color='#c52c42', zorder=10,
               label='Требуемое положение (2; 0.2)')
    ax.set(xlabel='t', ylabel='θ(t)', xlim=(0, 2.08))
    ax.legend(loc='upper left')

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.8), layout='constrained')
    figures.append(fig)
    title(fig, 'Построить график невязки от начальной скорости\n'
               'и показать приближения обоих методов')
    ss = np.linspace(*BRACKET, 101)
    residual = Residual(h)
    rr = [residual(float(s)) for s in ss]
    for ax, result, color, marker in ((axes[0], nr, '#1869b6', 'o'),
                                       (axes[1], br, '#d36a20', 'x')):
        ax.plot(ss, rr, color='#606871', linewidth=1.3, label='R(s)')
        ax.axhline(0, color='black', linewidth=0.8)
        xs, rs = np.array(result.history).T
        ax.scatter(xs, rs, color=color, marker=marker, s=48, zorder=4,
                   label='Приближения')
        ax.set(title=f'{result.method}: {result.calls} выз. R',
               xlabel='Начальная скорость s', ylabel='R(s) = θ(2; s) − 0.2',
               xlim=(-0.015, 0.54), ylim=(-0.225, 0.295))
        if result.method == 'Ньютон':
            ax.annotate('n = 0', (xs[0], rs[0]), xytext=(-42, 10),
                        textcoords='offset points', color=color)
            ax.annotate('', (xs[-1], rs[-1]), xytext=(xs[0], rs[0]),
                        arrowprops={'arrowstyle': '->', 'color': color,
                                    'linewidth': 2, 'linestyle': '--'})
            last_index = result.iterations
        else:
            # Поздние приближения сливаются около корня: подпишем первые три
            # и последнее, сохранив на графике все вычисленные приближения.
            offsets = ((18, 28), (-40, -25), (-55, 18))
            for i, ((s, r), offset) in enumerate(zip(result.history[:3], offsets), 1):
                ax.annotate(f'n = {i}', (s, r), xytext=offset,
                            textcoords='offset points', fontsize=8, color=color,
                            arrowprops={'arrowstyle': '-', 'color': color, 'linewidth': 0.8})
            last_index = result.iterations
        ax.scatter([result.s], [result.residual], s=110, marker='*',
                   color='#146647', edgecolors='white', linewidths=0.5, zorder=6)
        ax.annotate(f'n = {last_index}\ns = {result.s:.7f}',
                    (result.s, result.residual), xytext=(35, -90),
                    textcoords='offset points', color='#146647',
                    arrowprops={'arrowstyle': '->', 'color': '#146647'})
        ax.legend(loc='upper left')

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), layout='constrained')
    figures.append(fig)
    title(fig, 'Сопоставить численное и точное решения, показать изменение ошибки '
               'при уменьшении шага интегрирования')
    axes[0].plot(dense_t, exact_solution(dense_t), color='#252525',
                 label='Точное решение', linewidth=2.5)
    for step, result, _ in runs:
        t, y, _ = integrate(result.s, step)
        if step in (STEPS[0], STEPS[-1]):
            axes[0].plot(t, y, '--', marker='o', markerfacecolor='white',
                         markevery=max(1, len(t)//10), markersize=4,
                         label=f'Численное, h = {step:g}')
        if step in (0.4, 0.1, 0.025):
            axes[1].semilogy(t[1:], np.maximum(np.abs(y-exact_solution(t))[1:], 1e-17),
                             '.-', markersize=3, label=f'h = {step:g}')
    hs = np.array([row[0] for row in runs])
    errors = np.array([trajectory_error(result, step) for step, result, _ in runs])
    axes[2].loglog(hs, errors, 'o-', color='#1869b6', markersize=6)
    axes[2].set_xticks(hs, [f'{step:g}' for step in hs])
    axes[2].invert_xaxis()
    axes[0].set(title='Численное и точное решения', xlabel='t', ylabel='θ(t)')
    axes[1].set(title='Ошибка движения при разных шагах', xlabel='t',
                ylabel='|θ_h(t) − θ_точн(t)|')
    axes[2].set(title='Изменение ошибки при уменьшении шага',
                xlabel='Шаг интегрирования h (уменьшается →)',
                ylabel='Максимальная ошибка в узлах')
    axes[0].legend(loc='upper left')
    axes[1].legend(loc='lower left')
    return figures


def arrange_windows(figures, plt):
    """Два окна сверху, третье на всю ширину снизу, без перекрытия.

    На Windows учитываются панель задач и размеры рамок. Используется TkAgg.
    На других ОС доступная область получается через Tk.
    """
    try:
        # Сначала показать все окна: у скрытого окна Tk ещё не знает верные
        # размеры рамок, поэтому его геометрия может дать неверные отступы.
        for fig in figures:
            manager = fig.canvas.manager
            if getattr(manager, 'toolbar', None) is not None:
                manager.toolbar.pack_forget()
            manager.show()
        for fig in figures:
            fig.canvas.manager.window.update()
        first = figures[0].canvas.manager.window
        first.update_idletasks()
        left, top = 0, 0
        width, height = first.winfo_screenwidth(), first.winfo_screenheight()
        if __import__('sys').platform == 'win32':
            import ctypes
            from ctypes import wintypes
            rect = wintypes.RECT()
            if ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0):
                left, top = rect.left, rect.top
                width, height = rect.right-rect.left, rect.bottom-rect.top
        gap = 8
        cell_width = (width-3*gap)//2
        cell_height = (height-3*gap)//2
        for index, fig in enumerate(figures):
            manager = fig.canvas.manager
            window = manager.window
            window.update_idletasks()
            border_x = max(0, window.winfo_rootx()-window.winfo_x())
            title_y = max(0, window.winfo_rooty()-window.winfo_y())
            outer_width = width-2*gap if len(figures) == 3 and index == 2 else cell_width
            client_width = max(200, outer_width-2*border_x)
            client_height = max(160, cell_height-title_y-border_x)
            x = left+gap+(index % 2)*(cell_width+gap)
            y = top+gap+(index // 2)*(cell_height+gap)
            window.geometry(f'{client_width}x{client_height}+{x}+{y}')
        # Применить геометрию всех окон до общего цикла обработки событий.
        for fig in figures:
            fig.canvas.manager.window.update_idletasks()
            fig.canvas.draw_idle()
    except (AttributeError, RuntimeError) as exc:
        warnings.warn(f'Автоматическое размещение окон недоступно: {exc}', stacklevel=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--no-gui', action='store_true', help='Выполнить расчёт без графических окон')
    parser.add_argument('--save-plots', type=Path, metavar='DIR', help='Сохранить три графика в PNG')
    args = parser.parse_args()
    try:
        import matplotlib
        # TkAgg даёт отдельные обычные окна и предсказуемое размещение на Windows.
        matplotlib.use('Agg' if args.no_gui else 'TkAgg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit('Нужны matplotlib и Tk: python -m pip install numpy matplotlib. '
                         'Без графических окон используйте --no-gui.') from exc
    runs, tolerances, study = compute_experiments()
    print_report(runs, tolerances, study)
    if args.no_gui and args.save_plots is None:
        return
    figures = make_figures(plt, runs)
    if args.save_plots is not None:
        args.save_plots.mkdir(parents=True, exist_ok=True)
        for index, fig in enumerate(figures, 1):
            fig.savefig(args.save_plots / f'4.1_graph_{index}.png', dpi=160)
        print(f'\nГрафики сохранены в: {args.save_plots.resolve()}')
    if args.no_gui:
        plt.close('all')
    else:
        arrange_windows(figures, plt)
        plt.show(block=True)


if __name__ == '__main__':
    main()
