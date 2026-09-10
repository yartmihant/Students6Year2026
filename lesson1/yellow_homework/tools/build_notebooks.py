"""Build the three self-contained teaching notebooks (standard library only)."""
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def md(s):
    return dict(cell_type="markdown", metadata={}, source=textwrap.dedent(s).strip() + "\n")


def code(s):
    return dict(cell_type="code", metadata={}, execution_count=None, outputs=[],
                source=textwrap.dedent(s).strip() + "\n")


COMMON = r'''
import time
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML, Markdown
from IPython import get_ipython
from scipy.optimize import brentq

get_ipython().run_line_magic('matplotlib', 'inline')
started = time.perf_counter()
plt.rcParams.update({'figure.dpi': 90, 'font.size': 10,
                     'axes.grid': True, 'animation.embed_limit': 50})
COLORS = plt.cm.viridis(np.linspace(0.08, 0.92, 10))

def step(rhs, state, h, method='RK4'):
    """Явный Эйлер или классический RK4 для автономной системы."""
    k1 = rhs(state)
    if method == 'Euler':
        return state + h*k1
    if method != 'RK4':
        raise ValueError(method)
    k2 = rhs(state + h*k1/2)
    k3 = rhs(state + h*k2/2)
    k4 = rhs(state + h*k3)
    return state + h*(k1 + 2*k2 + 2*k3 + k4)/6

def integrate(rhs, initial, duration, h, method='RK4'):
    # Последний момент точно равен duration; фактический шаг не больше h.
    n = int(np.ceil(duration/h))
    t = np.linspace(0, duration, n+1)
    y = np.empty((n+1, len(initial)))
    y[0] = initial
    for j in range(n):
        y[j+1] = step(rhs, y[j], t[j+1]-t[j], method)
    return t, y

def sample(t, y, times):
    """Интерполяция только для кадров; ошибки считаются на расчётной сетке."""
    return np.column_stack([np.interp(times, t, y[:, j]) for j in range(y.shape[1])])

def table(headers, rows):
    escaped = lambda values: [str(value).replace('|', '&#124;') for value in values]
    display(Markdown('| ' + ' | '.join(escaped(headers)) + ' |\n| ' +
                     ' | '.join(['---']*len(headers)) + ' |\n' +
                     '\n'.join('| ' + ' | '.join(escaped(row)) + ' |' for row in rows)))

def show_animation(fig, update, frames, filename):
    # Кадры встроены: работают без ffmpeg, сети и дополнительных виджетов.
    movie = FuncAnimation(fig, update, frames=frames, interval=70, blit=False)
    html = movie.to_jshtml(default_mode='once')
    # Matplotlib использует внешний шрифт для значков; заменяем локальным текстом.
    html = re.sub(r'<link\s+rel="stylesheet"\s+href="[^"]+">', '', html)
    icons = {'fa-minus': '−', 'fa-plus': '+', 'fa-fast-backward': '⏮',
             'fa-backward': '◀|', 'fa-step-backward': '|◀', 'fa-play': '▶',
             'fa-play fa-flip-horizontal': '◀', 'fa-step-forward': '▶|', 'fa-pause': 'Ⅱ',
             'fa-stop': '■', 'fa-forward': '|▶', 'fa-fast-forward': '⏭',
             'fa-refresh': '↻'}
    html = re.sub(r'<i class="fa ([^"]+)"[^>]*></i>',
                  lambda match: icons.get(match[1], match[1].removeprefix('fa-')), html)
    plt.close(fig)
    out = Path('animations')
    out.mkdir(exist_ok=True)
    (out/filename).write_text('<!doctype html><meta charset="utf-8">' + html, encoding='utf-8')
    display(HTML(html))
'''


def save(name, cells):
    for i, cell in enumerate(cells):
        cell['id'] = f'cell-{i:03d}'
    doc = dict(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.13.0'}}, nbformat=4, nbformat_minor=5)
    (ROOT/name).write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding='utf-8')


save('1.1_pendulum_yellow.ipynb', [
md(r'''
# 1.1. Математический маятник — 🟢 + 🟡

**Цель:** отделить ошибку линейной модели от ошибки интегрирования и увидеть
зависимость периода от амплитуды. Все ячейки выполняются сверху вниз; анимации
имеют кнопки воспроизведения и ползунок. Параметры можно менять в ячейках ниже.

Состояние $y=(\theta,\dot\theta)$, углы в радианах. Берём $m=L=1$, $g=9.81$;
$\omega_0=\sqrt{g/L}$ и $T_0=2\pi/\omega_0$. Начальная угловая скорость нулевая.
Линейная модель: $\ddot\theta=-\omega_0^2\theta$; нелинейная:
$\ddot\theta=-\omega_0^2\sin\theta$.

Для линейной модели $U=mgL\theta^2/2$, для нелинейной $U=mgL(1-\cos\theta)$;
в обоих случаях $K=mL^2\dot\theta^2/2$. Смешивать эти определения при проверке
сохранения энергии нельзя. Разделы 1–2 закрывают базовый уровень, остальные — жёлтый.
'''), code(COMMON),
code(r'''
g, L, m = 9.81, 1.0, 1.0
w0 = np.sqrt(g/L)
T0 = 2*np.pi/w0

def pendulum(length=L, nonlinear=True):
    return lambda y: np.array([y[1], -g/length*(np.sin(y[0]) if nonlinear else y[0])])

def energy(y, nonlinear=True, length=L):
    kinetic = m*length**2*y[:, 1]**2/2
    potential = m*g*length*(1-np.cos(y[:, 0]) if nonlinear else y[:, 0]**2/2)
    return kinetic, potential
'''),
md(r'''
## 1. Линейная модель: точное решение и сходимость

При $\theta(0)=A$, $\dot\theta(0)=0$ точное решение равно
$\theta=A\cos\omega_0t$, $\dot\theta=-A\omega_0\sin\omega_0t$.
Считаем до $5T_0$. Ошибка — максимум $|\theta_h-\theta_{exact}|$ по всей сетке.
При делении шага пополам ожидается примерно двукратное уменьшение ошибки Эйлера
и шестнадцатикратное для RK4 в асимптотическом режиме.
'''),
code(r'''
A = np.deg2rad(10)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
rows, errors = [], {}
for method in ['Euler', 'RK4']:
    errors[method] = []
    for n in [100, 200, 400, 800]:
        t, y = integrate(pendulum(nonlinear=False), [A, 0], 5*T0, T0/n, method)
        err = np.max(np.abs(y[:, 0]-A*np.cos(w0*t)))
        errors[method].append(err)
        rows.append([method, f'T₀/{n}', f'{err:.3e}'])
    t, y = integrate(pendulum(nonlinear=False), [A, 0], 5*T0, T0/200, method)
    K, U = energy(y, nonlinear=False)
    ax[0].plot(t/T0, y[:, 0], label=method)
    ax[1].plot(y[:, 0], y[:, 1], label=method)
    ax[2].plot(t/T0, K, label=f'K {method}')
    ax[2].plot(t/T0, U, '--', label=f'U {method}')
tt = np.linspace(0, 5*T0, 2000)
ax[0].plot(tt/T0, A*np.cos(w0*tt), 'k:', label='Точно')
ax[1].plot(A*np.cos(w0*tt), -A*w0*np.sin(w0*tt), 'k:', label='Точно')
for a, xlabel, ylabel in zip(ax, ['t / T₀', 'θ, рад', 't / T₀'], ['θ, рад', 'dθ/dt, рад/с', 'Энергия, Дж']):
    a.set(xlabel=xlabel, ylabel=ylabel); a.legend(fontsize=8)
fig.tight_layout(); display(fig); plt.close(fig)
table(['Метод', 'Шаг', 'max |Δθ|, рад'], rows)
assert errors['RK4'][-2]/errors['RK4'][-1] > 12
assert errors['Euler'][-1] < errors['Euler'][0]
'''),
md(r'''
## 2. Нелинейная модель: период и критерий точности линейного приближения

Независимый эталон для $0<A<\pi$:
$T(A)=4\sqrt{L/g}\,K(\sin^2(A/2))$. Здесь `scipy.special.ellipk` принимает
**параметр** $m=\sin^2(A/2)$, а не модуль $\sin(A/2)$.

Численный период равен четырём временам первого пересечения $\theta=0$ сверху
при старте из покоя. После обнаружения смены знака уточняем событие внутри шага:
ищем корень $\theta(\mathrm{RK4}(y_j,\delta t))=0$. Это устраняет ошибку простого
округления к узлу сетки; интегрирование до события всё ещё имеет свою ошибку.

Слово «достаточно» определим явно: относительная ошибка периода
$(T(A)-T_0)/T(A)\leq1\%$. Также посчитаем порог для 5%.
'''),
code(r'''
from scipy.special import ellipk

def exact_period(amplitude, length=L):
    return 4*np.sqrt(length/g)*ellipk(np.sin(amplitude/2)**2)

def measured_period(amplitude, h):
    rhs = pendulum()
    t, y = integrate(rhs, [amplitude, 0], exact_period(amplitude)/2, h)
    j = np.flatnonzero((y[:-1, 0] > 0) & (y[1:, 0] <= 0))[0]
    dt = brentq(lambda dt: step(rhs, y[j], dt)[0], 0, t[j+1]-t[j], xtol=1e-14)
    return 4*(t[j]+dt)

degrees = np.linspace(2, 160, 45)
amplitudes = np.deg2rad(degrees)
periods = np.array([measured_period(a, T0/250) for a in amplitudes])
fig, ax = plt.subplots(figsize=(7.5, 3.5))
ax.plot(degrees, exact_period(amplitudes)/T0, label='Эллиптический интеграл')
ax.plot(degrees, periods/T0, '.', label='RK4, событие θ=0')
ax.axhline(1, color='k', ls='--', label='Линейная модель')
ax.set(xlabel='Начальная амплитуда, градусы', ylabel='T(A) / T₀'); ax.legend()
display(fig); plt.close(fig)
thresholds = [np.rad2deg(brentq(lambda a: 1-T0/exact_period(a)-eps, 1e-6, 3.0)) for eps in [.01, .05]]
display(Markdown(f'**Ответ:** при выбранном критерии линейная модель предсказывает период '
                 f'с ошибкой до 1% примерно при **A ≤ {thresholds[0]:.2f}°**, '
                 f'до 5% — при **A ≤ {thresholds[1]:.2f}°**. '
                 'При росте амплитуды нелинейный период растёт.'))
assert np.max(np.abs(periods/exact_period(amplitudes)-1)) < 2e-6
'''),
md(r'''
## 3. Раздельно: ошибка модели и ошибка шага при A = 90°

Первый график сравнивает линейную модель с нелинейным расчётом на мелкой сетке.
Второй показывает только ошибку нелинейного периода относительно точной формулы.
Уменьшение шага не исправляет ошибку линеаризации. Дополнительно проверяем
обмен кинетической и потенциальной энергией в нелинейной системе.
'''),
code(r'''
A = np.pi/2
TA = exact_period(A)
rows, period_errors = [], []
ns = np.array([25, 50, 100, 200])
for n in ns:
    numerical = measured_period(A, T0/n)
    error = abs(numerical-TA)
    period_errors.append(error)
    rows.append([f'T₀/{n}', f'{numerical:.10f}', f'{error:.3e}'])
table(['Шаг', 'Численный период, с', '|T_h − T(A)|, с'], rows)
t, yn = integrate(pendulum(), [A, 0], 4*TA, T0/500)
yl = np.column_stack([A*np.cos(w0*t), -A*w0*np.sin(w0*t)])
K, U = energy(yn)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
ax[0].plot(t/T0, yl[:, 0], label='Линейная'); ax[0].plot(t/T0, yn[:, 0], label='Нелинейная')
ax[0].set(xlabel='t / T₀', ylabel='θ, рад'); ax[0].legend()
ax[1].loglog(T0/ns, period_errors, 'o-'); ax[1].set(xlabel='Шаг, с', ylabel='Ошибка периода, с')
ax[1].set_xticks(T0/ns, [f'{h:.3f}' for h in T0/ns]); ax[1].xaxis.set_minor_formatter(plt.NullFormatter())
for vals, label in [(K, 'K'), (U, 'U'), (K+U, 'K+U')]:
    ax[2].plot(t/T0, vals, label=label)
ax[2].set(xlabel='t / T₀', ylabel='Энергия, Дж'); ax[2].legend()
fig.tight_layout(); display(fig); plt.close(fig)
display(Markdown(f'При A=90° ошибка линейного периода составляет **{100*(1-T0/TA):.2f}%**. '
                 f'Ошибка RK4 при самом мелком проверенном шаге — **{period_errors[-1]:.2e} с**.'))
assert period_errors[-2]/period_errors[-1] > 12
assert np.max(np.abs((K+U)/(K[0]+U[0])-1)) < 1e-7
'''),
md(r'''
## 4. Анимация: линейный и нелинейный маятники с фазовыми следами

Слева геометрическое положение груза; справа $(\theta,\dot\theta)$.
Оба движения показаны в один физический момент времени. Для линейного маятника
используется точное решение, для нелинейного — RK4 из предыдущей ячейки.
'''),
code(r'''
frames = np.linspace(0, 3*TA, 101)
nonlinear = sample(t, yn, frames)
linear = np.column_stack([A*np.cos(w0*frames), -A*w0*np.sin(w0*frames)])
fig, ax = plt.subplots(1, 2, figsize=(8, 3.4), dpi=75)
rods, trails, dots = [], [], []
for color, label in [('tab:blue', 'Линейный'), ('tab:orange', 'Нелинейный')]:
    rods.append(ax[0].plot([], [], 'o-', color=color, label=label)[0])
    trails.append(ax[1].plot([], [], color=color, alpha=.7)[0])
    dots.append(ax[1].plot([], [], 'o', color=color, label=label)[0])
ax[0].set(xlim=(-1.2, 1.2), ylim=(-1.2, .3), aspect='equal', xlabel='x, м', ylabel='y, м')
ax[1].set(xlim=(-1.8, 1.8), ylim=(-5.3, 5.3), xlabel='θ, рад', ylabel='dθ/dt, рад/с')
ax[0].legend(); title = fig.suptitle(''); fig.tight_layout()
def update(j):
    for state, rod, trail, dot in zip([linear, nonlinear], rods, trails, dots):
        angle = state[j, 0]
        rod.set_data([0, L*np.sin(angle)], [0, -L*np.cos(angle)])
        trail.set_data(state[:j+1, 0], state[:j+1, 1])
        dot.set_data([state[j, 0]], [state[j, 1]])
    title.set_text(f'A = 90°, t = {frames[j]:.2f} с')
show_animation(fig, update, len(frames), '1.1_comparison.html')
'''),
md(r'''
## 5. Десять длин и соизмеримые частоты

Для всех маятников $A=20°$, но $\omega_{0,i}=n_i\omega_b$, $n_i=10,\ldots,19$,
$\omega_b=0.4$ с⁻¹ и $L_i=g/\omega_{0,i}^2$.
Используем **нелинейную** модель: при одинаковой амплитуде множитель поправки
к периоду одинаков, поэтому соизмеримость сохраняется.
Общее время повторения $T_{sync}=4K(\sin^2(A/2))/\omega_b$.
Чтобы не пропускать быстрые колебания, здесь 241 кадр.
'''),
code(r'''
ensemble_A = np.deg2rad(20)
harmonics = np.arange(10, 20)
wb = .4
lengths = g/(harmonics*wb)**2
sync = 4*ellipk(np.sin(ensemble_A/2)**2)/wb
frames = np.linspace(0, sync, 241)
states = []
for length in lengths:
    ti, yi = integrate(pendulum(length), [ensemble_A, 0], sync, .002)
    states.append(sample(ti, yi, frames))
fig, ax = plt.subplots(figsize=(9, 3.4), dpi=75)
pivots = np.arange(10)*.6
rods = [ax.plot([], [], 'o-', color=c)[0] for c in COLORS]
for x, n in zip(pivots, harmonics):
    ax.text(x, .06, f'n={n}', ha='center', fontsize=8)
ax.set(xlim=(-.5, pivots[-1]+.5), ylim=(-.72, .17), xlabel='x, м', ylabel='y, м', aspect='equal')
title = ax.set_title(''); fig.tight_layout()
def update(j):
    for i, rod in enumerate(rods):
        theta = states[i][j, 0]
        rod.set_data([pivots[i], pivots[i]+lengths[i]*np.sin(theta)], [0, -lengths[i]*np.cos(theta)])
    title.set_text(f'Соизмеримые частоты, t / T_sync = {frames[j]/sync:.3f}')
show_animation(fig, update, len(frames), '1.1_lengths.html')
assert max(abs(s[-1, 0]-ensemble_A) for s in states) < 1e-6
'''),
md('''
## 6. Одинаковые длины, разные амплитуды

Все длины равны 1 м, начальные углы от 10° до 150°, старт из покоя.
Точки подвеса разнесены только для читаемости. Даже без взаимодействия и
диссипации фазы расходятся: большие амплитуды имеют большие периоды.
'''),
code(r'''
angles = np.deg2rad(np.linspace(10, 150, 10))
frames = np.linspace(0, 4*T0, 121)
states = []
for angle in angles:
    ti, yi = integrate(pendulum(), [angle, 0], frames[-1], T0/400)
    states.append(sample(ti, yi, frames))
fig, axes = plt.subplots(2, 5, figsize=(9, 4.5), dpi=75, sharex=True, sharey=True)
rods = []
for panel, angle, color in zip(axes.flat, angles, COLORS):
    rods.append(panel.plot([], [], 'o-', color=color)[0])
    panel.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.05), aspect='equal', title=f'A={np.rad2deg(angle):.0f}°')
    panel.set_xticks([-1, 0, 1]); panel.set_yticks([-1, 0, 1])
title = fig.suptitle(''); fig.tight_layout(rect=(0, 0, 1, .92))
def update(j):
    for i, rod in enumerate(rods):
        theta = states[i][j, 0]
        rod.set_data([0, L*np.sin(theta)], [0, -L*np.cos(theta)])
    title.set_text(f'Одинаковая длина, разные амплитуды; t = {frames[j]:.2f} с')
show_animation(fig, update, len(frames), '1.1_amplitudes.html')
'''),
md('''
## Выводы и оценка сложности

- RK4 сходится к нелинейной модели; переход к малому шагу не делает её линейной.
- При критерии ошибки периода 1% допустимая амплитуда около 23°; ответ зависит от допуска.
- Соизмеримые частоты можно получить подбором длин, если начальные амплитуды одинаковы.
  При разных амплитудах поправки к периодам различаются.
- Самые содержательные этапы: определение периода по событию и разделение двух ошибок.
  Самая трудоёмкая техническая часть: три анимации и синхронизация панелей.

**Оценка для студента после первого семинара:** 6–10 часов при знакомстве с Python,
10–16 часов при первом опыте NumPy/Matplotlib/Jupyter. Это экспертная оценка,
а не измеренное время работы студента. Расчёт периода через эллиптический интеграл —
дополнительный эталон для проверки; для выполнения задания достаточно сходимости численного периода.
'''),
code("print(f'Все численные проверки пройдены. Полное выполнение: {time.perf_counter()-started:.1f} с.')")
])

save('1.2_ballistics_yellow.ipynb', [
md(r'''
# 1.2. Баллистика в 2D — 🟢 + 🟡

Состояние $y=(x,z,v_x,v_z)$, ось $z$ направлена вверх. Во всех опытах $m=1$ кг,
$g=9.81$ м/с², $v_0=20$ м/с, старт и приземление на $z=0$.
Сначала проверяем Эйлера и RK4 на линейном сопротивлении, затем исследуем
$\dot{\mathbf v}=(0,-g)-k|\mathbf v|\mathbf v$; здесь $k$ имеет размерность м⁻¹.
Сила сопротивления равна $-mk|\mathbf v|\mathbf v$.

Все ячейки самостоятельного ноутбука выполняются последовательно. Анимации
встроены в результаты и дополнительно сохраняются в `animations/`.
'''), code(COMMON),
code(r'''
g, mass, v0 = 9.81, 1.0, 20.0

def initial(angle):
    a = np.deg2rad(angle)
    return np.array([0., 0., v0*np.cos(a), v0*np.sin(a)])

def linear_rhs(tau):
    return lambda s: np.array([s[2], s[3], -s[2]/tau, -g-s[3]/tau])

def linear_exact(t, angle, tau):
    vx, vz = initial(angle)[2:]
    q = -np.expm1(-np.asarray(t)/tau)
    return np.column_stack([tau*vx*q, tau*(vz+g*tau)*q-g*tau*t,
                            vx*(1-q), (vz+g*tau)*(1-q)-g*tau])

def quadratic_rhs(k):
    def rhs(s):
        speed = np.hypot(s[2], s[3])
        return np.array([s[2], s[3], -k*speed*s[2], -g-k*speed*s[3]])
    return rhs

def flight(angle, k=0., h=.02, method='RK4', rhs=None):
    """Первое возвращение на землю; последний неполный шаг уточняется по z=0."""
    rhs = quadratic_rhs(k) if rhs is None else rhs
    ts, ys = [0.], [initial(angle)]
    for _ in range(int(60/h)+1):
        candidate = step(rhs, ys[-1], h, method)
        if candidate[1] < 0:
            # Для положительного угла пересечение должно быть после подъёма.
            if ys[-1][1] <= 0:
                raise ValueError('Слишком крупный шаг: подъём не разрешён')
            dt = brentq(lambda dt: step(rhs, ys[-1], dt, method)[1], 0, h, xtol=1e-13)
            landing = step(rhs, ys[-1], dt, method)
            assert abs(landing[1]) < 1e-9
            landing[1] = 0.0
            ts.append(ts[-1]+dt); ys.append(landing)
            return np.array(ts), np.array(ys)
        ts.append(ts[-1]+h); ys.append(candidate)
    raise RuntimeError('Не найден момент падения за 60 с')
'''),
md(r'''
## 1. Проверка на линейном сопротивлении

Обозначим $q=1-e^{-t/\tau}$. Тогда $x=\tau v_{x0}q$,
$z=\tau(v_{z0}+g\tau)q-g\tau t$, $v_x=v_{x0}e^{-t/\tau}$,
$v_z=(v_{z0}+g\tau)e^{-t/\tau}-g\tau$.

Сравниваем решения до **точного** момента падения при $\tau=2$ с, $\alpha=45°$.
Ошибка — максимум евклидовой ошибки положения на сетке. Энергия
$E=K+U=m|v|^2/2+mgz$ должна убывать: $dE/dt=-m|v|^2/\tau$.
'''),
code(r'''
tau, angle = 2., 45.
tf = brentq(lambda t: linear_exact(np.array([t]), angle, tau)[0, 1], .01, 10)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
rows, errors = [], {}
for method in ['Euler', 'RK4']:
    errors[method] = []
    for h in [.16, .08, .04, .02]:
        t, y = integrate(linear_rhs(tau), initial(angle), tf, h, method)
        exact = linear_exact(t, angle, tau)
        err = np.max(np.linalg.norm(y[:, :2]-exact[:, :2], axis=1))
        errors[method].append(err)
        rows.append([method, f'{t[1]:.5f}', f'{err:.3e}'])
    t, y = integrate(linear_rhs(tau), initial(angle), tf, .08, method)
    ax[0].plot(y[:, 0], y[:, 1], label=method)
    K, U = mass*np.sum(y[:, 2:]**2, axis=1)/2, mass*g*y[:, 1]
    for vals, label, ls in [(K, 'K', '-'), (U, 'U', '--'), (K+U, 'E', ':')]:
        ax[1].plot(t, vals, ls, label=f'{label} {method}')
exact = linear_exact(np.linspace(0, tf, 400), angle, tau)
ax[0].plot(exact[:, 0], exact[:, 1], 'k--', label='Точно')
for drag_tau in [.5, 2., 10.]:
    ti, yi = flight(angle, h=.02, rhs=linear_rhs(drag_tau))
    ax[2].plot(yi[:, 0], yi[:, 1], label=f'τ={drag_tau} с')
ti, yi = flight(angle, k=0)
ax[2].plot(yi[:, 0], yi[:, 1], 'k--', label='Без сопротивления')
for a in [ax[0], ax[2]]:
    a.set(xlabel='x, м', ylabel='z, м'); a.legend(fontsize=8)
ax[1].set(xlabel='t, с', ylabel='Энергия, Дж'); ax[1].legend(fontsize=8, ncol=2)
fig.tight_layout(); display(fig); plt.close(fig)
table(['Метод', 'Фактический шаг, с', 'max ошибка положения, м'], rows)
assert errors['RK4'][-2]/errors['RK4'][-1] > 12
assert errors['Euler'][-1] < errors['Euler'][0]
'''),
md(r'''
## 2. Квадратичное сопротивление и максимум дальности

Перебираем углы от 5° до 85° с шагом 1°. Окрестность лучшего угла уточняем
перебором с шагом 0.05°. Это два конечных перебора, не непрерывная оптимизация.
При $k=0$ эталон $R=v_0^2\sin(2\alpha)/g$ даёт максимум при 45°.

Момент приземления ищется внутри последнего шага с помощью повторного шага RK4
длиной $\delta t$ и скалярного поиска корня высоты. Простое сохранение первого
подземного узла внесло бы дополнительную ошибку порядка шага.
'''),
code(r'''
angles = np.arange(5., 86.)
ks = [0., .01, .03, .08]

def scan(k, h):
    ranges = np.array([flight(a, k, h)[1][-1, 0] for a in angles])
    best = angles[np.argmax(ranges)]
    fine = np.arange(best-1, best+1.0001, .05)
    rf = np.array([flight(a, k, h)[1][-1, 0] for a in fine])
    j = np.argmax(rf)
    return ranges, float(fine[j]), float(rf[j])

fig, ax = plt.subplots(figsize=(8, 4))
maxima = {}
for k in ks:
    ranges, optimum, distance = scan(k, .02)
    maxima[k] = (optimum, distance)
    line, = ax.plot(angles, ranges, label=f'k={k:g} м⁻¹')
    ax.plot(optimum, distance, 'o', color=line.get_color())
    ax.annotate(f'{optimum:.2f}°', (optimum, distance), xytext=(6, 5), textcoords='offset points')
ax.set(xlabel='Угол запуска, градусы', ylabel='Дальность, м'); ax.legend()
fig.tight_layout(); display(fig); plt.close(fig)
table(['k, м⁻¹', 'Лучший угол сетки, °', 'Дальность, м'],
      [[k, f'{a:.2f}', f'{r:.5f}'] for k, (a, r) in maxima.items()])
assert abs(maxima[0][0]-45) < 1e-8
for a in [15, 45, 75]:
    tfree, yfree = flight(a, 0, .07)
    assert abs(yfree[-1, 0]-v0**2*np.sin(2*np.deg2rad(a))/g) < 1e-9
    assert abs(tfree[-1]-2*v0*np.sin(np.deg2rad(a))/g) < 1e-10
'''),
md(r'''
## 3. Сходимость: не путать шаг времени с шагом перебора углов

Для $k=0.03$ повторим весь поиск при четырёх временных шагах.
Дополнительно сравним дальность **при фиксированном** угле 40° с расчётом
на шаге 0.00125 с. Это численный эталон, не точное решение.
Неизменность лучшего угла сама по себе не доказывает сходимости:
угловая сетка различает только 0.05°.
'''),
code(r'''
k_selected = .03
ref_t, ref_y = flight(40, k_selected, .00125)
rows, range_errors = [], []
for h in [.08, .04, .02, .01]:
    _, opt, distance = scan(k_selected, h)
    ti, yi = flight(40, k_selected, h)
    err = abs(yi[-1, 0]-ref_y[-1, 0])
    range_errors.append(err)
    rows.append([h, f'{opt:.2f}', f'{distance:.8f}', f'{ti[-1]:.9f}', f'{err:.3e}'])
table(['h, с', 'Лучший угол, °', 'R_max, м', 't_падения при 40°, с', '|R_h(40°)−R_ref|, м'], rows)
assert range_errors[-2]/range_errors[-1] > 10

# Асимметрия: уточняем вершину по v_z=0 тем же приёмом.
t, y = flight(45, k_selected, .01)
rhs = quadratic_rhs(k_selected)
j = np.flatnonzero((y[:-1, 3] > 0) & (y[1:, 3] <= 0))[0]
dt = brentq(lambda dt: step(rhs, y[j], dt)[3], 0, t[j+1]-t[j])
apex = step(rhs, y[j], dt)
t_up, t_down = t[j]+dt, t[-1]-t[j]-dt
landing_angle = np.rad2deg(np.arctan2(-y[-1, 3], y[-1, 2]))
K, U = mass*np.sum(y[:, 2:]**2, axis=1)/2, mass*g*y[:, 1]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(y[:, 0], y[:, 1], label='k=0.03, запуск 45°')
ax[0].axvline(apex[0], color='k', ls=':', label='Вершина')
ax[0].axvline(y[-1, 0]/2, color='gray', ls='--', label='Половина дальности')
ax[0].set(xlabel='x, м', ylabel='z, м'); ax[0].legend(fontsize=8)
for vals, label in [(K, 'K'), (U, 'U'), (K+U, 'E')]: ax[1].plot(t, vals, label=label)
ax[1].set(xlabel='t, с', ylabel='Энергия, Дж'); ax[1].legend()
fig.tight_layout(); display(fig); plt.close(fig)
display(Markdown(f'**Ответ:** для исследованных коэффициентов сопротивления оптимальный угол '
                 f'уменьшается от 45° до {maxima[.08][0]:.2f}°. '
                 f'При k=0.03 и запуске 45° подъём длится {t_up:.3f} с, '
                 f'спуск — {t_down:.3f} с; вершина находится на '
                 f'{100*apex[0]/y[-1, 0]:.1f}% дальности. '
                 f'Угол падения к горизонту {landing_angle:.1f}°, то есть спуск круче подъёма. '
                 'Горизонтальная скорость падает, поэтому траектория теряет зеркальную симметрию.'))
assert np.max(np.diff(K+U)) < 1e-7
'''),
md(r'''
## 4. Полёт со скоростью и силами

Стрелки имеют разные явно заданные масштабы: 0.12 м рисунка на 1 м/с для скорости,
0.20 м рисунка на 1 Н для сил. Сравнивать длины стрелок скорости и сил физически нельзя.
Гравитация постоянна; сопротивление направлено против скорости.
'''),
code(r'''
frames = np.linspace(0, t[-1], 91)
motion = sample(t, y, frames)
fig, ax = plt.subplots(figsize=(8, 3.8), dpi=75)
trail, = ax.plot([], [], color='tab:blue')
body, = ax.plot([], [], 'o', color='black')
arrows = [ax.quiver([0], [0], [0], [0], angles='xy', scale_units='xy', scale=1,
                    color=color, label=label) for color, label in
          [('tab:blue', 'v: 0.12 м / (м/с)'), ('tab:red', 'mg: 0.20 м / Н'), ('tab:green', 'Fсопр: 0.20 м / Н')]]
ax.axhline(0, color='k'); ax.set(xlim=(-4, y[-1, 0]+5), ylim=(-4, y[:, 1].max()+5),
                               aspect='equal', xlabel='x, м', ylabel='z, м')
ax.legend(loc='upper right', fontsize=8); title = ax.set_title(''); fig.tight_layout()
def update(j):
    s = motion[j]; v = s[2:]
    vectors = [.12*v, .20*np.array([0, -mass*g]), -.20*mass*k_selected*np.linalg.norm(v)*v]
    trail.set_data(motion[:j+1, 0], motion[:j+1, 1]); body.set_data([s[0]], [s[1]])
    for arrow, vector in zip(arrows, vectors):
        arrow.set_offsets(s[:2].reshape(1, 2)); arrow.set_UVC([vector[0]], [vector[1]])
    title.set_text(f't = {frames[j]:.2f} с, |v| = {np.linalg.norm(v):.2f} м/с')
show_animation(fig, update, len(frames), '1.2_vectors.html')
'''),
md('''
## 5. Одновременный запуск: без сопротивления и с ним

Десять углов от 10° до 80°, одна скорость, единая физическая шкала времени
и одинаковые пределы координат на обеих панелях. После падения тело перестаёт
двигаться, в точке падения остаётся крестик. Цвет соответствует углу.
'''),
code(r'''
launch_angles = np.linspace(10, 80, 10)
flights = [[flight(a, k, .01) for a in launch_angles] for k in [0, k_selected]]
frames = np.linspace(0, max(ti[-1] for panel in flights for ti, _ in panel), 111)
motions = [[sample(ti, yi, frames) for ti, yi in panel] for panel in flights]
fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=75, sharex=True, sharey=True)
artists = []
for panel, k in zip(ax, [0, k_selected]):
    group = []
    for a, color in zip(launch_angles, COLORS):
        trail, = panel.plot([], [], color=color, lw=1, label=f'{a:.0f}°')
        dot, = panel.plot([], [], 'o', color=color, ms=4)
        mark, = panel.plot([], [], 'x', color=color, ms=7)
        group.append((trail, dot, mark))
    artists.append(group)
    panel.set(title=f'k={k:g} м⁻¹', xlim=(-1, 44), ylim=(-1, 22), aspect='equal', xlabel='x, м')
    panel.axhline(0, color='k')
ax[0].set_ylabel('z, м'); ax[1].legend(ncol=2, fontsize=7, loc='upper right')
title = fig.suptitle(''); fig.tight_layout()
def update(j):
    for p in range(2):
        for i, (trail, dot, mark) in enumerate(artists[p]):
            ti, yi = flights[p][i]; motion = motions[p][i]
            trail.set_data(motion[:j+1, 0], motion[:j+1, 1])
            if frames[j] >= ti[-1]:
                dot.set_data([], []); mark.set_data([yi[-1, 0]], [0])
            else:
                dot.set_data([motion[j, 0]], [motion[j, 1]]); mark.set_data([], [])
    title.set_text(f'Одновременный запуск, t = {frames[j]:.2f} с')
show_animation(fig, update, len(frames), '1.2_ensemble.html')
'''),
md('''
## Выводы и оценка сложности

- Дальность и оптимальный угол уменьшаются с ростом сопротивления в рассмотренном диапазоне.
- Сопротивление нарушает симметрию: вершина смещается к дальней половине дальности,
  нисходящая ветвь круче, скорость при падении меньше начальной.
- Уточнение момента падения необходимо для осмысленного сравнения временных шагов.
  Точность угла дополнительно ограничивает дискретный перебор.
- Наиболее сложные места: обработка события приземления и остановка отдельных тел
  в общей анимации. Поиск корня здесь нужен лишь внутри уже найденного шага.

**Оценка для студента:** 7–12 часов со знанием Python; 12–18 часов при первом опыте
численного моделирования и анимаций. Это ориентир преподавателю, не результат хронометража.
'''),
code("print(f'Все численные проверки пройдены. Полное выполнение: {time.perf_counter()-started:.1f} с.')")
])

save('1.3_orbits_yellow.ipynb', [
md(r'''
# 1.3. Орбитальное движение в 2D — 🟢 + 🟡

Безразмерные переменные, гравитационный параметр $\mu=1$, масса пробного тела 1.
Состояние $s=(x,y,v_x,v_y)$, ускорение $-\mathbf r/r^3$.
Сначала проверяем круговую орбиту, затем исследуем эллипсы с **одинаковой**
большой полуосью $a=1$ и эксцентриситетами от 0 до 0.8.
Все тела независимы: между собой не взаимодействуют, центр неподвижен.

Все ячейки выполняются сверху вниз. Для воспроизведения анимаций используйте ▶.
'''), code(COMMON),
code(r'''
mu, a = 1., 1.
T = 2*np.pi*np.sqrt(a**3/mu)

def gravity(s):
    r = np.linalg.norm(s[:2])
    return np.r_[s[2:], -mu*s[:2]/r**3]

def orbit_initial(e):
    return np.array([a*(1-e), 0., 0., np.sqrt(mu/a*(1+e)/(1-e))])

def invariants(s):
    energy = np.sum(s[:, 2:]**2, axis=1)/2-mu/np.linalg.norm(s[:, :2], axis=1)
    angular_momentum = s[:, 0]*s[:, 3]-s[:, 1]*s[:, 2]
    return energy, angular_momentum

def exact_orbit(t, e):
    # Уравнение Кеплера E-e sin E=M решаем Ньютоном; e<=0.8 в опытах ниже.
    M = np.asarray(t)*np.sqrt(mu/a**3)
    E = M.copy()
    for _ in range(40):
        correction = (E-e*np.sin(E)-M)/(1-e*np.cos(E))
        E -= correction
        if np.max(np.abs(correction)) < 1e-13:
            break
    assert np.max(np.abs(E-e*np.sin(E)-M)) < 1e-11
    factor = np.sqrt(mu/a**3)/(1-e*np.cos(E))
    return np.column_stack([a*(np.cos(E)-e), a*np.sqrt(1-e**2)*np.sin(E),
                            -a*np.sin(E)*factor, a*np.sqrt(1-e**2)*np.cos(E)*factor])
'''),
md(r'''
## 1. Круговая орбита: проверка двух методов

При $(x,y,v_x,v_y)=(1,0,0,1)$ точное положение равно $(\cos t,\sin t)$.
Удельная энергия $E=|v|^2/2-1/r=-1/2$, момент импульса $L_z=xv_y-yv_x=1$.
Сравниваем орбиты и инварианты за два оборота. В таблице максимум ошибки положения
относительно точного решения, а не только ошибка в последнем узле.
'''),
code(r'''
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
rows, errors = [], {}
for method in ['Euler', 'RK4']:
    errors[method] = []
    for n in [100, 200, 400, 800]:
        t, s = integrate(gravity, orbit_initial(0), 2*T, T/n, method)
        err = np.max(np.linalg.norm(s[:, :2]-exact_orbit(t, 0)[:, :2], axis=1))
        errors[method].append(err); rows.append([method, f'T/{n}', f'{err:.3e}'])
    t, s = integrate(gravity, orbit_initial(0), 2*T, T/400, method)
    E, J = invariants(s)
    ax[0].plot(s[:, 0], s[:, 1], label=method)
    ax[1].plot(t/T, E, label=method); ax[2].plot(t/T, J, label=method)
exact = exact_orbit(np.linspace(0, T, 600), 0)
ax[0].plot(exact[:, 0], exact[:, 1], 'k--', label='Точно')
ax[0].plot(0, 0, 'k*'); ax[0].set(xlabel='x', ylabel='y', aspect='equal')
ax[1].axhline(-.5, color='k', ls='--'); ax[1].set(xlabel='t / T', ylabel='E')
ax[2].axhline(1, color='k', ls='--'); ax[2].set(xlabel='t / T', ylabel='Lz')
for panel in ax: panel.legend()
fig.tight_layout(); display(fig); plt.close(fig)
table(['Метод', 'Шаг', 'max ошибка положения'], rows)
assert errors['RK4'][-2]/errors['RK4'][-1] > 12
assert errors['Euler'][-1] < errors['Euler'][0]
'''),
md(r'''
## 2. Эллипсы с общей большой полуосью

Старт из перицентра: $r_p=a(1-e)$, $v_p=\sqrt{(1+e)/(a(1-e))}$.
Аналитическая форма: $x=a(\cos E-e)$, $y=a\sqrt{1-e^2}\sin E$,
где эксцентрическая аномалия $E$ связана со временем уравнением Кеплера
$E-e\sin E=\sqrt{1/a^3}\,t$. Для всех эксцентриситетов $T=2\pi a^{3/2}$.

Численный период измерим по следующему пересечению $y=0$ снизу вверх при $x>0$,
уточнив событие внутри шага RK4. Это измерение периода не использует возвращение
по готовой формуле времени. При $e=0$ пересечение служит той же меткой фазы.
'''),
code(r'''
eccentricities = [0., .3, .6, .8]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
rows = []
for e in eccentricities:
    t, s = integrate(gravity, orbit_initial(e), 1.15*T, T/4000)
    indices = np.flatnonzero((s[:-1, 1] < 0) & (s[1:, 1] >= 0) & (s[1:, 0] > 0))
    j = indices[0]
    dt = brentq(lambda dt: step(gravity, s[j], dt)[1], 0, t[j+1]-t[j], xtol=1e-14)
    measured = t[j]+dt
    exact = exact_orbit(t, e)
    line, = ax[0].plot(s[:, 0], s[:, 1], label=f'e={e:g}, RK4')
    ax[0].plot(exact[:, 0], exact[:, 1], '--', color=line.get_color(), alpha=.6)
    ax[1].plot(t/T, np.linalg.norm(s[:, 2:], axis=1), label=f'e={e:g}')
    shape_error = np.max(np.abs((s[:, 0]+a*e)**2/a**2+s[:, 1]**2/(a*a*(1-e*e))-1))
    rows.append([e, f'{measured:.9f}', f'{abs(measured/T-1):.3e}', f'{shape_error:.3e}'])
    assert abs(measured/T-1) < 2e-6
    assert shape_error < 2e-5
ax[0].plot(0, 0, 'k*'); ax[0].set(xlabel='x', ylabel='y', aspect='equal', title='Сплошная: RK4; штриховая: точно')
ax[1].set(xlabel='t / T', ylabel='|v|', title='Распределение скорости')
for panel in ax: panel.legend(fontsize=8)
fig.tight_layout(); display(fig); plt.close(fig)
table(['e', 'Численный период', '|T_h/T−1|', 'max невязка уравнения эллипса'], rows)
print(f'Аналитический период для всех e: {T:.9f}')
'''),
md(r'''
## 3. Ошибка возвращения через точный период и уменьшение шага

Считаем $\|\mathbf r_h(T)-\mathbf r(0)\|$ строго в момент $T$, без интерполяции
между соседними узлами: шаг равен $T/N$. Для Эйлера и RK4 используем одну сетку.
Ошибка возвращения объединяет ошибки формы и фазы; сама по себе она не заменяет
проверку траектории и периода из предыдущего раздела.
'''),
code(r'''
ns = np.array([250, 500, 1000, 2000])
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
rows, return_errors = [], {}
for panel, method in zip(ax, ['Euler', 'RK4']):
    for e in eccentricities:
        errs = []
        for n in ns:
            ti, si = integrate(gravity, orbit_initial(e), T, T/n, method)
            err = np.linalg.norm(si[-1, :2]-si[0, :2])
            errs.append(err)
            rows.append([method, e, int(n), f'{err:.3e}'])
        return_errors[(method, e)] = errs
        panel.loglog(T/ns, errs, 'o-', label=f'e={e:g}')
    panel.set(title=method, xlabel='Шаг h', ylabel='Ошибка возвращения'); panel.legend()
    panel.set_xticks(T/ns, [f'{h:.4f}' for h in T/ns]); panel.xaxis.set_minor_formatter(plt.NullFormatter())
fig.tight_layout(); display(fig); plt.close(fig)
table(['Метод', 'e', 'Шагов за T', 'Ошибка возвращения'], rows)
for e in eccentricities:
    errs = return_errors[('RK4', e)]
    assert errs[-1] < errs[0]/100
    assert errs[-2]/errs[-1] > 12
table(['e', 'v в перицентре', 'v в апоцентре', 'v_p / v_a', 'r_p / v_p'],
      [[e, f'{np.sqrt((1+e)/(1-e)):.3f}', f'{np.sqrt((1-e)/(1+e)):.3f}',
        f'{(1+e)/(1-e):.1f}', f'{(1-e)**1.5/np.sqrt(1+e):.4f}'] for e in eccentricities])
display(Markdown('**Ответ:** в перицентре тело движется быстрее, в апоцентре медленнее. '
                 'При e=0.8 отношение скоростей равно 9. Локальное время прохождения '
                 'перицентра rₚ/vₚ уменьшается примерно в 15 раз относительно круга. '
                 'Поэтому одинаковое число шагов за период хуже разрешает вытянутую орбиту; '
                 'при постоянном шаге его следует выбирать по быстрому движению у перицентра.'))
'''),
md('''
## 4. Эйлер и RK4: две панели, один момент времени

Выбран e=0.6, одинаковый шаг T/2000. Штриховой линией показана точная форма,
пустым кружком — точное положение в текущий момент. След позволяет увидеть
ошибку формы, разность положений — ошибку фазы. Масштаб панелей общий.
'''),
code(r'''
e_selected = .6
frames = np.linspace(0, T, 111)
motion = []
for method in ['Euler', 'RK4']:
    ti, si = integrate(gravity, orbit_initial(e_selected), T, T/2000, method)
    motion.append(sample(ti, si, frames))
reference = exact_orbit(frames, e_selected)
outline = exact_orbit(np.linspace(0, T, 800), e_selected)
fig, ax = plt.subplots(1, 2, figsize=(9, 4), dpi=75, sharex=True, sharey=True)
artists = []
all_positions = np.vstack([s[:, :2] for s in motion]+[outline[:, :2]])
lo, hi = all_positions.min(axis=0)-.15, all_positions.max(axis=0)+.15
for panel, method in zip(ax, ['Euler', 'RK4']):
    panel.plot(outline[:, 0], outline[:, 1], 'k--', label='Точный эллипс')
    panel.plot(0, 0, 'k*')
    trail, = panel.plot([], [], color='tab:blue')
    dot, = panel.plot([], [], 'o', color='tab:blue', label=method)
    exact_dot, = panel.plot([], [], 'o', mfc='none', mec='tab:red', ms=8, label='Точное положение')
    artists.append((trail, dot, exact_dot))
    panel.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), aspect='equal', xlabel='x', ylabel='y', title=method)
    panel.legend(fontsize=8)
title = fig.suptitle(''); fig.tight_layout()
def update(j):
    for s, (trail, dot, exact_dot) in zip(motion, artists):
        trail.set_data(s[:j+1, 0], s[:j+1, 1]); dot.set_data([s[j, 0]], [s[j, 1]])
        exact_dot.set_data([reference[j, 0]], [reference[j, 1]])
    title.set_text(f'e=0.6, h=T/2000, t/T={frames[j]/T:.3f}')
show_animation(fig, update, len(frames), '1.3_methods.html')
'''),
md('''
## 5. Десять орбит с разными эксцентриситетами

Одинаковая большая полуось a=1 и общий период T. Все тела стартуют из своих
перицентров при t=0. Для движения используется RK4 с шагом T/4000, для формы —
аналитические эллипсы. Быстрые участки находятся около общего притягивающего центра.
'''),
code(r'''
es = np.linspace(0, .8, 10)
frames = np.linspace(0, T, 121)
motions = []
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=75)
dots = []
for e, color in zip(es, COLORS):
    ti, si = integrate(gravity, orbit_initial(e), T, T/4000)
    motions.append(sample(ti, si, frames))
    outline = exact_orbit(np.linspace(0, T, 800), e)
    ax.plot(outline[:, 0], outline[:, 1], color=color, alpha=.4, lw=1)
    dots.append(ax.plot([], [], 'o', color=color, label=f'e={e:.2f}')[0])
ax.plot(0, 0, 'k*', ms=12)
ax.set(xlim=(-1.95, 1.15), ylim=(-1.15, 1.15), aspect='equal', xlabel='x', ylabel='y')
ax.legend(loc='center left', bbox_to_anchor=(1.01, .5), fontsize=8)
title = ax.set_title('Ten elliptical orbits'); fig.tight_layout()
def update(j):
    for s, dot in zip(motions, dots): dot.set_data([s[j, 0]], [s[j, 1]])
    title.set_text(f'Общий период, t/T = {frames[j]/T:.3f}')
show_animation(fig, update, len(frames), '1.3_ensemble.html')
'''),
md(r'''
## 6. Равные площади за равные времена

Разделим период на 12 равных интервалов. Сектор — область между двумя
радиусами-векторами и дугой орбиты, а не круговой сектор с постоянным радиусом.
Теоретическая площадь каждого $S_i=\pi a^2\sqrt{1-e^2}/12$, поскольку
$dS/dt=L_z/2$. Проверяем площади по **численной траектории RK4** формулой
площади многоугольника с вершиной в центре притяжения. Затем показываем постепенное
закрашивание этих же секторов; справа численная скорость и общий указатель времени.
'''),
code(r'''
from matplotlib.patches import Polygon

sectors, points_per_sector = 12, 600
ti, si = integrate(gravity, orbit_initial(e_selected), T, T/(sectors*points_per_sector))
expected_area = np.pi*a*a*np.sqrt(1-e_selected**2)/sectors
areas = []
for k in range(sectors):
    part = si[k*points_per_sector:(k+1)*points_per_sector+1, :2]
    areas.append(abs(np.sum(part[:-1, 0]*part[1:, 1]-part[1:, 0]*part[:-1, 1]))/2)
table(['Сектор', 'Площадь по RK4', 'Относительная ошибка к теории'],
      [[j+1, f'{area:.8f}', f'{abs(area/expected_area-1):.3e}'] for j, area in enumerate(areas)])
assert max(abs(np.array(areas)/expected_area-1)) < 2e-5

fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=75)
ax[0].plot(si[:, 0], si[:, 1], 'k-', lw=1)
ax[0].plot(0, 0, 'k*')
patches = []
for k in range(sectors):
    patch = Polygon(np.zeros((3, 2)), facecolor=plt.cm.tab20(k), alpha=.65)
    ax[0].add_patch(patch); patches.append(patch)
radius, = ax[0].plot([], [], 'k-', lw=1)
body, = ax[0].plot([], [], 'ko')
ax[0].set(xlim=(-1.8, .6), ylim=(-1, 1), aspect='equal', xlabel='x', ylabel='y')
speed = np.linalg.norm(si[:, 2:], axis=1)
ax[1].plot(ti/T, speed, color='tab:blue')
for boundary in np.linspace(0, 1, sectors+1): ax[1].axvline(boundary, color='gray', lw=.5)
cursor = ax[1].axvline(0, color='red'); dot, = ax[1].plot([], [], 'ro')
ax[1].set(xlabel='t / T', ylabel='|v|', xlim=(0, 1))
title = fig.suptitle(''); fig.tight_layout()
indices = np.linspace(0, len(ti)-1, 121).round().astype(int)
def update(j):
    idx = indices[j]
    for k, patch in enumerate(patches):
        start, stop = k*points_per_sector, min(idx, (k+1)*points_per_sector)
        patch.set_visible(stop > start)
        if stop > start: patch.set_xy(np.vstack([[0, 0], si[start:stop+1, :2], [0, 0]]))
    radius.set_data([0, si[idx, 0]], [0, si[idx, 1]])
    body.set_data([si[idx, 0]], [si[idx, 1]])
    cursor.set_xdata([ti[idx]/T]*2); dot.set_data([ti[idx]/T], [speed[idx]])
    title.set_text(f'e=0.6, t/T={ti[idx]/T:.3f}; Δt=T/12, S≈{expected_area:.4f}')
show_animation(fig, update, len(indices), '1.3_sectors.html')
'''),
md('''
## Выводы и оценка сложности

- При фиксированной большой полуоси период не зависит от эксцентриситета.
- Скорость максимальна в перицентре; с ростом e на этот участок приходится всё меньше времени.
  Постоянный шаг нужно уменьшать, что подтверждают ошибки возвращения.
- Явный Эйлер заметно искажает орбиту и инварианты. RK4 существенно точнее на тех же шагах,
  но это наблюдение на нескольких оборотах, а не утверждение о долговременной точности.
- Равные площади за равные времена согласуются с постоянством момента импульса.
- Самые сложные этапы: согласованные начальные условия при разных e, независимый
  расчёт периода, уравнение Кеплера и правильное построение заметённых секторов.

**Оценка для студента:** 9–15 часов со знанием Python и механики; 15–24 часа при
первом опыте NumPy/Matplotlib. Это наиболее трудоёмкая из трёх работ, главным образом
из-за аналитического эталона и третьей анимации. Уравнение Кеплера нужно здесь для
точного положения во времени; для наложения одной только формы достаточно параметрического эллипса.
'''),
code("print(f'Все численные проверки пройдены. Полное выполнение: {time.perf_counter()-started:.1f} с.')")
])

print('Built 3 notebooks in', ROOT)
