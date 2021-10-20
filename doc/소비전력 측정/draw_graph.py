import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
import numpy as np

x = np.arange(32)
cpu = [ 1, 0.5, 0.6, 1.1, 0.3, 0.7,0.5,0.6, 1, 0.7, 0.9, 1.1,
        12.05555556, 8.711111111, 8.244444444, 9.216666667, 13.84444444, 10.88888889,
        7, 9.683333333, 9.138888889, 11.16111111, 11.2, 13.61111111, 11.00555556,
        10.5, 14.15555556, 6.533333333, 10.69444444, 10.00055451, 9.45784545, 11.23]

gpu = [72, 78, 56, 64, 78, 81, 48, 77, 66, 98, 67, 67, 52, 63, 84, 66]
gpu = [24,24,25,24,24,24,25,24,24,24,24,45,72, 88, 79, 74, 88, 91, 79, 81, 96, 88, 83, 78, 88, 82, 94, 89, 82, 99, 89, 91]

xnew = np.linspace(x.min(), x.max(), 300)
spl = make_interp_spline(x, cpu, k=3)
cpu_smooth = spl(xnew)

spl = make_interp_spline(x, gpu, k=3)
gpu_smooth = spl(xnew)

plt.plot(cpu_smooth, label='cpu')
plt.plot(gpu_smooth, label='gpu')
plt.xlabel('second (s)')
plt.ylabel('power (Watts)')
plt.grid()
plt.legend()
plt.ylim(0,120)
plt.savefig('savefig_default.png')
