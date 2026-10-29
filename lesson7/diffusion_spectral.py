"""Расчётные функции альтернативной лекции. Код также включается внутрь ноутбука."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, eye
from scipy.sparse.linalg import splu
from scipy.linalg import eigh
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML

METHODS = {'EE': 0., 'IE': 1., 'CN': .5}
COLORS = {'EE': '#2878b5', 'IE': '#d47d17', 'CN': '#248267'}
LABELS = {'EE': 'Явный Эйлер', 'IE': 'Неявный Эйлер', 'CN': 'Кранк–Николсон'}


def amplification(z, method):
    """R(z), z = tau * lambda; полюса возвращают inf/nan."""
    z = np.asarray(z, dtype=complex)
    theta = METHODS[method]
    with np.errstate(divide='ignore', invalid='ignore'):
        return (1+(1-theta)*z)/(1-theta*z)


def grid_spectrum(N=64, kappa=1.):
    h = 1/N
    x = np.arange(N)*h  # x=1 не дублируется
    m = np.arange(N)
    lam = -4*kappa/h**2*np.sin(np.pi*m/N)**2
    return x, h, lam


def periodic_matrix(N=64, kappa=1.):
    """Циклическая трёхдиагональная матрица: есть два угловых элемента."""
    a = kappa*N*N
    A = diags([np.full(N-1,a), np.full(N,-2*a), np.full(N-1,a)],
              [-1,0,1], format='lil')
    A[0,-1] = A[-1,0] = a
    return A.tocsc()


def profile(x, kind='smooth'):
    # Гладкий периодический аналог гауссова импульса: все производные на концах согласованы.
    if kind == 'smooth':
        return np.exp(-8*(1-np.cos(2*np.pi*(x-.5))))
    if kind == 'box':
        return ((x >= .375) & (x < .625)).astype(float)
    raise ValueError(kind)


def semidiscrete_exact(u0, times, kappa=1.):
    """Точный эталон системы U'=A U, а не непрерывного уравнения."""
    lam = grid_spectrum(len(u0), kappa)[2]
    return np.fft.ifft(np.exp(np.outer(np.asarray(times),lam))*np.fft.fft(u0), axis=1).real


def periodic_solve(u0, T, tau, method, kappa=1., unsafe=False, rannacher=False):
    """Равномерные шаги до одного T; фактический tau <= запрошенного.

    Сохраняется каждый слой. Неявные методы решают циклическую систему,
    FFT используется только для независимого эталона.
    """
    N = len(u0)
    if T <= 0 or tau <= 0 or kappa <= 0:
        raise ValueError('T, tau, kappa должны быть положительными')
    steps = int(np.ceil(T/tau-1e-12))
    if steps > 100000:
        raise ValueError('Слишком много шагов для лекционной демонстрации')
    tau = T/steps
    r = kappa*tau*N*N
    if method == 'EE' and r > .5+1e-12 and not unsafe:
        raise ValueError(f'Для EE нарушено r <= 1/2: r={r:g}')
    if rannacher and method != 'CN':
        raise ValueError('Сглаживающий старт предусмотрен для CN')
    A = periodic_matrix(N,kappa)
    I = eye(N, format='csc')
    theta = METHODS[method]
    B = I+(1-theta)*tau*A
    solver = splu(I-theta*tau*A).solve if theta else None
    U = np.empty((steps+1,N)); U[0] = u0
    for n in range(steps):
        if n == 0 and rannacher:
            half = splu(I-.5*tau*A).solve
            U[n+1] = half(half(U[n]))
        else:
            rhs = B @ U[n]
            U[n+1] = solver(rhs) if solver else rhs
        if not np.isfinite(U[n+1]).all() or np.max(abs(U[n+1])) > 1e100:
            raise FloatingPointError(f'Чрезмерный рост при t={(n+1)*tau:g}')
    times = np.linspace(0,T,steps+1)
    mean = U.mean(axis=1)
    return dict(t=times,u=U,tau=tau,r=r,steps=steps,
                heat=mean,maximum=np.max(abs(U),axis=1),
                l2=np.sqrt(np.mean((U-mean[:,None])**2,axis=1)))


def panel(fn, **controls):
    """Общий интерфейс: изменить параметры и нажать «Обновить».

    Без ipywidgets используются обычные вызовы функций, уже присутствующие ниже.
    """
    try:
        import ipywidgets as w
    except ImportError:
        print('Виджеты не установлены. Измените параметры обычного вызова в следующей ячейке.')
        return
    widget = w.interactive(fn, {'manual': True, 'manual_name': 'Обновить'}, **controls)
    display(widget)


def experiment1(N=32, m=1, parts='Обе части'):
    x = np.arange(N)/N
    m = int(m)  # допускаются номера вне основного диапазона: это алиасы
    xx = np.linspace(0,1,max(1200,40*(abs(m)+N)))
    fig, ax = plt.subplots(1,2,figsize=(13,3.6))
    for part,color,label in [(np.cos,'#2878b5','Re: cos'),(np.sin,'#d47d17','Im: sin')]:
        if parts == 'Только Re' and part is np.sin: continue
        if parts == 'Только Im' and part is np.cos: continue
        ax[0].plot(xx,part(2*np.pi*m*xx),color=color,alpha=.45)
        ax[0].plot(x,part(2*np.pi*m*x),'o',color=color,label=label,ms=4)
    ax[1].plot(xx,np.cos(2*np.pi*m*xx),label=f'm = {m}',lw=2)
    ax[1].plot(xx,np.cos(2*np.pi*(m+N)*xx),alpha=.45,label=f'm + N = {m+N}')
    ax[1].plot(x,np.cos(2*np.pi*m*x),'ko',ms=4,label='Совпадающие отсчёты')
    for a in ax:
        a.set(xlim=(0,1),ylim=(-1.2,1.2),xlabel='x',ylabel='Значение');a.legend(fontsize=9)
    ax[0].set_title(f'N={N}, m={m}, θ=2π·{m}/{N}')
    ax[1].set_title('Алиасинг: различие только между узлами')
    fig.tight_layout();plt.show()
    alias_error = np.max(abs(np.exp(2j*np.pi*m*x)-np.exp(2j*np.pi*(m+N)*x)))
    print(f'Ошибка совпадения в узлах: {alias_error:.2e}. ' +
          ('На сетке есть мода Найквиста m=N/2.' if N%2==0 else 'N нечётно: отдельной моды Найквиста нет.'))


def experiment2(real=-3., imag=0.):
    z = complex(real,imag)
    re,im = np.meshgrid(np.linspace(-5,3,400),np.linspace(-4,4,400))
    Z = re+1j*im
    fig,axes = plt.subplots(2,3,figsize=(14,7))
    tt=np.linspace(0,12,600);nn=np.arange(13)
    for col,method in enumerate(METHODS):
        R=amplification(Z,method)
        axes[0,col].contourf(re,im,(abs(R)<=1).astype(int),levels=[-.5,.5,1.5],colors=['#f7f7f7','#c6e2d8'])
        axes[0,col].contour(re,im,abs(R),levels=[1],colors=['#248267'])
        axes[0,col].plot(z.real,z.imag,'ro');axes[0,col].axvline(0,color='grey',lw=.6)
        axes[0,col].axhline(0,color='grey',lw=.6)
        axes[0,col].set(xlabel='Re z',ylabel='Im z',title=LABELS[method],aspect='equal')
        r=complex(amplification(z,method))
        if not np.isfinite(r):
            axes[1,col].text(.1,.5,'Полюс R(z): шаг не определён',transform=axes[1,col].transAxes)
            continue
        yy=r**nn
        axes[1,col].plot(tt,np.exp(z*tt).real,'k--',label='Re точного')
        axes[1,col].plot(nn,yy.real,'o-',color=COLORS[method],label='Re численного')
        if abs(z.imag)>1e-12:
            axes[1,col].plot(tt,np.exp(z*tt).imag,':',color='grey',label='Im точного')
            axes[1,col].plot(nn,yy.imag,'s:',color=COLORS[method],label='Im численного')
        # Общий симметричный логарифмический масштаб: рост EE не прячет остальные методы.
        axes[1,col].set_yscale('symlog',linthresh=1)
        axes[1,col].set(ylim=(-1e12,1e12),xlabel='t / τ = n',ylabel='y, symlog',title=f'R={r:.3g}, |R|={abs(r):.3g}')
        axes[1,col].set_yticks([-1e12,-1e6,-1,0,1,1e6,1e12])
        axes[1,col].minorticks_off()
        axes[1,col].legend(fontsize=8)
    fig.tight_layout();plt.show()
    print('Начальное значение y₀=1; τ=1, λ=z. Зелёный фон: |R(z)|≤1. Для Re z>0 точное решение тоже растёт.')


def experiment3(a=10.):
    aa=np.linspace(0,40,500)
    fig,ax=plt.subplots(1,2,figsize=(12,3.8))
    ax[0].plot(aa,np.exp(-aa),'k--',label='exp(−a)')
    for method in ['IE','CN']:
        ax[0].plot(aa,amplification(-aa,method).real,label=LABELS[method],color=COLORS[method])
    ax[0].axhline(0,color='grey',lw=.6);ax[0].axvline(a,color='grey',ls=':')
    ax[0].set(xlabel='a = μτ',ylabel='Множитель за шаг',ylim=(-1.1,1.1));ax[0].legend()
    n=np.arange(11)
    ax[1].plot(n,np.exp(-a*n),'k--',label='Точное')
    for method in ['IE','CN']:
        ax[1].plot(n,amplification(-a,method).real**n,'o-',label=method,color=COLORS[method])
    ax[1].set(xlabel='Номер шага',ylabel='yₙ',ylim=(-1.1,1.1),title=f'a={a:g}');ax[1].legend()
    fig.tight_layout();plt.show()


def experiment4(N=64, kappa=1.):
    x,h,lam=grid_spectrum(N,kappa)
    modes=[1,N//8,N//2]
    T=.02/kappa;times=np.linspace(0,T,200)
    fig,ax=plt.subplots(1,3,figsize=(14,3.8))
    m=np.arange(N)
    ax[0].plot(m,lam,'o',ms=3);ax[0].set(xlabel='Номер m в DFT',ylabel='λₘ',title='Спектр периодического оператора')
    for mode in modes:
        ax[1].semilogy(times,np.exp(lam[mode]*times),label=f'm={mode}')
    ax[1].set(ylim=(1e-12,1.1),xlabel='Время',ylabel='Амплитуда');ax[1].legend()
    initial=.3+sum(np.cos(2*np.pi*mode*x)/3 for mode in modes)
    for t,u in zip([0,.1*h*h/kappa,h*h/kappa,T],semidiscrete_exact(initial,[0,.1*h*h/kappa,h*h/kappa,T],kappa)):
        ax[2].plot(x,u,label=f't={t:.3g}')
    ax[2].set(xlabel='x',ylabel='Температура',ylim=(-.8,1.5));ax[2].legend(fontsize=8)
    fig.tight_layout();plt.show()
    print(f'h={h:g}; max |λ|={max(abs(lam)):.4g}; мода m=0 не затухает.')
    print('m>N/2 соответствует отрицательным частотам m−N; частота не растёт монотонно с индексом DFT.')


def experiment5(N=64):
    x,h,lam=grid_spectrum(N)
    u0=profile(x)+1e-5*(-1.)**np.arange(N)
    T=30*h*h  # для r=.25,.5,.6 получается целое число шагов
    fig,ax=plt.subplots(1,3,figsize=(14,3.8))
    runs={}
    for r in [.25,.5,.6]:
        res=periodic_solve(u0,T,r*h*h,'EE',unsafe=r>.5);runs[r]=res
        ax[0].plot(x,res['u'][-1],label=f'r={r}')
        ax[1].semilogy(res['t'],res['maximum'],label=f'r={r}')
        ax[2].semilogy(res['t'],res['l2'],label=f'r={r}')
    exact=semidiscrete_exact(u0,[T])[0]
    ax[0].plot(x,exact,'k--',label='Эталон')
    ax[0].set(xlabel='x',ylabel='U',title=f'Общее время T={T:.4g}')
    ax[1].set(xlabel='Время',ylabel='max |U|')
    ax[2].set(xlabel='Время',ylabel='‖U − среднее‖₂,h')
    for a in ax:a.legend(fontsize=8)
    fig.tight_layout();plt.show()
    fig,ax=plt.subplots(figsize=(8,2.8))
    for n in [0,1,2]:
        ax.plot(x[:12],(-1.)**(np.arange(N)+n)[:12],'o-',label=f'Слой n={n}',alpha=.65)
    ax.set(xlabel='x (фрагмент сетки)',ylabel='Мода Найквиста',ylim=(-1.2,1.2),title='r=1/2: G(π)=−1, затухания нет');ax.legend()
    fig.tight_layout();plt.show()
    print('Максимумы при общем T:',{r:float(res['maximum'][-1]) for r,res in runs.items()})
    return runs


def experiment6(N=64):
    x,h,_=grid_spectrum(N);u0=profile(x);T=.01
    exact=semidiscrete_exact(u0,[T])[0]
    configs=[('EE',.4*h*h),('IE',.4*h*h),('IE',T/10),('IE',T)]
    fig,ax=plt.subplots(1,2,figsize=(12,3.8))
    ax[0].plot(x,exact,'k--',lw=2,label='Полудискретный эталон')
    for method,tau in configs:
        res=periodic_solve(u0,T,tau,method)
        label=f'{method}: {res["steps"]} шаг., r={res["r"]:.2g}'
        ax[0].plot(x,res['u'][-1],label=label)
        ax[1].plot(x,res['u'][-1]-exact,label=label)
        print(f'{label}; E∞={max(abs(res["u"][-1]-exact)):.4e}')
    ax[0].set(xlabel='x',ylabel='Температура',ylim=(-.05,1.05));ax[0].legend(fontsize=8)
    ax[1].set(xlabel='x',ylabel='Ошибка при T=0.01');ax[1].legend(fontsize=8)
    fig.tight_layout();plt.show()


def convergence_data(N=64):
    x,h,_=grid_spectrum(N);u0=profile(x);T=.02
    # Все методы сравниваются на одних шагах, но неустойчивый EE не запускается.
    steps_list=[8,16,32,64,128,256,512]
    exact=semidiscrete_exact(u0,[T])[0]
    records=[]
    for steps in steps_list:
        tau=T/steps;r=tau/h**2
        for method in METHODS:
            if method=='EE' and r>.5:
                records.append((method,tau,r,steps,np.nan,np.nan))
                continue
            res=periodic_solve(u0,T,tau,method)
            err=max(abs(res['u'][-1]-exact))
            drift=max(abs(res['heat']-res['heat'][0]))
            records.append((method,tau,r,steps,err,drift))
    return records


def experiment7(N=128, r=20.):
    x,h,_=grid_spectrum(N);u0=profile(x,'box');tau=r*h*h;T=4*tau
    refs=semidiscrete_exact(u0,[tau,T])
    fig,ax=plt.subplots(1,2,figsize=(12,3.8))
    for method,start,label in [('IE',False,'IE'),('CN',False,'CN'),('CN',True,'CN + Раннахер')]:
        res=periodic_solve(u0,T,tau,method,rannacher=start)
        for a,index,ref in zip(ax,[1,4],refs):a.plot(x,res['u'][index],label=label)
    for a,t,ref in zip(ax,[tau,T],refs):
        a.plot(x,ref,'k--',label='Эталон');a.set(xlim=(.25,.75),ylim=(-.25,1.25),xlabel='x',ylabel='U',title=f't={t:.4g}');a.legend(fontsize=9)
    fig.tight_layout();plt.show()
    records=convergence_data(64)
    fig,ax=plt.subplots(figsize=(7,3.5))
    for method in ['IE','CN']:
        data=[row for row in records if row[0]==method]
        ax.loglog([v[1] for v in data],[v[4] for v in data],'o-',label=method,color=COLORS[method])
        print(method,'временной порядок по двум самым мелким шагам:',np.log2(data[-2][4]/data[-1][4]))
    ax.set(xlabel='τ',ylabel='E∞ относительно exp(tA)U₀',title='Гладкий периодический импульс, N=64, T=0.02');ax.legend()
    fig.tight_layout();plt.show()


def boundary_problem(kind,N=80,kappa=.2,transfer=2.,T=8.,tau=.02):
    """Конечные объёмы, c_v=1. Робин: сопротивление h/(2κ)+1/α."""
    h=1/N;x=(np.arange(N)+.5)*h
    interior=np.full(N-1,kappa/h)
    if kind=='Нейман':
        cL=cR=0.;TL=TR=0.;equilibrium=np.full(N,(1+.5*np.sin(np.pi*x)).mean())
    elif kind=='Дирихле':
        cL=cR=2*kappa/h;TL=1.;TR=0.;equilibrium=1-x
    elif kind=='Робин':
        cL=cR=1/(h/(2*kappa)+1/transfer);TL=TR=0.;equilibrium=np.zeros(N)
    else:raise ValueError(kind)
    face=np.r_[cL,interior,cR]
    A=diags([interior/h,-(face[:-1]+face[1:])/h,interior/h],[-1,0,1],format='csc')
    b=np.zeros(N);b[0]=cL*TL/h;b[-1]=cR*TR/h
    u0=1+.5*np.sin(np.pi*x)
    steps=int(np.ceil(T/tau));tau=T/steps;times=np.linspace(0,T,steps+1)
    solve=splu(eye(N,format='csc')-tau*A).solve
    U=np.empty((steps+1,N));U[0]=u0;integral=np.zeros(steps+1)
    for n in range(steps):
        U[n+1]=solve(U[n]+tau*b)
        qL=cL*(TL-U[n+1,0]);qR=cR*(U[n+1,-1]-TR)
        integral[n+1]=integral[n]+tau*(qL-qR)
    heat=h*U.sum(axis=1)
    values,vectors=eigh(A.toarray())
    # Независимый от шага времени эталон той же пространственной системы.
    reference=equilibrium+vectors@(np.exp(values*T)*(vectors.T@(u0-equilibrium)))
    return dict(x=x,t=times,u=U,heat=heat,balance=heat-heat[0]-integral,
                integral=integral,equilibrium=equilibrium,reference=reference)


def experiment8(transfer=2.):
    fig,ax=plt.subplots(1,3,figsize=(14,3.8))
    data={}
    for a,kind in zip(ax,['Нейман','Дирихле','Робин']):
        res=boundary_problem(kind,transfer=transfer);data[kind]=res
        for t in [0,.2,1,8]:
            i=np.argmin(abs(res['t']-t));a.plot(res['x'],res['u'][i],label=f't={res["t"][i]:g}')
        a.plot(res['x'],res['equilibrium'],'k--',label='Стационар')
        a.plot(res['x'],res['reference'],'k:',label='Эталон при T')
        a.set(xlabel='x',ylabel='u',ylim=(-.05,1.6),title=kind);a.legend(fontsize=8)
        print(kind,'невязка баланса:',max(abs(res['balance'])),
              'ошибка при T:',max(abs(res['u'][-1]-res['reference'])))
    fig.tight_layout();plt.show()
    fig,ax=plt.subplots(1,3,figsize=(14,3.4))
    for kind,res in data.items():
        ax[0].plot(res['t'],res['heat'],label=kind)
        line,=ax[1].plot(res['t'],res['heat']-res['heat'][0],label=kind)
        ax[1].plot(res['t'],res['integral'],'--',color=line.get_color())
        ax[2].plot(res['t'],res['balance'],label=kind)
    ax[0].set(xlabel='t',ylabel='H = h ΣU',title='Полное тепло')
    ax[1].set(xlabel='t',ylabel='ΔH и интеграл потоков',title='Сплошная: ΔH; штрих: приток − отток')
    ax[2].set(xlabel='t',ylabel='Невязка баланса')
    for a in ax:a.legend(fontsize=8)
    fig.tight_layout();plt.show()
    return data


def experiment9(N=64):
    records=convergence_data(N)
    print('Метод      τ           r     шаги       E∞           дрейф H')
    for method,tau,r,steps,error,drift in records:
        if np.isnan(error):
            print(f'{method:4} {tau:11.4g} {r:8.3f} {steps:5}   пропуск: r > 1/2')
        else:print(f'{method:4} {tau:11.4g} {r:8.3f} {steps:5} {error:12.4e} {drift:12.3e}')
    fig,ax=plt.subplots(1,2,figsize=(12,3.8))
    for method in METHODS:
        data=[v for v in records if v[0]==method and np.isfinite(v[4])]
        ax[0].loglog([v[1] for v in data],[v[4] for v in data],'o-',label=method,color=COLORS[method])
        ax[1].loglog([v[3] for v in data],[v[4] for v in data],'o-',label=method,color=COLORS[method])
    ax[0].set(xlabel='τ',ylabel='E∞',title='Общий профиль и общий момент T=0.02')
    ax[1].set(xlabel='Число шагов (не время выполнения)',ylabel='E∞',title='У IE/CN один шаг дороже')
    for a in ax:a.legend()
    fig.tight_layout();plt.show()
    return records


def decay_animation(N=64):
    x,h,lam=grid_spectrum(N);times=np.linspace(0,.02,65)
    u0=.3+.5*np.cos(2*np.pi*x)+.25*np.cos(8*np.pi*x)+.15*(-1.)**np.arange(N)
    exact=semidiscrete_exact(u0,times)
    fig,ax=plt.subplots(figsize=(9,3.5));line,=ax.plot(x,exact[0],'o-',ms=3)
    ax.set(xlim=(0,1),ylim=(-.65,1.3),xlabel='x',ylabel='u');title=ax.set_title('')
    def update(i):
        line.set_ydata(exact[i]);title.set_text(f'Точное затухание полудискретных мод, t={times[i]:.4f}')
        return line,title
    animation=FuncAnimation(fig,update,frames=len(times),interval=80)
    plt.close(fig)
    return HTML(animation.to_jshtml())
