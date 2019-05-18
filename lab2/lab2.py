from math import cos
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

Y_LEFT = 0

Y1_DEF = 0

DEF_P = 1
STEP_DEF = 0.001
ITER_NUMB_DEF = 1000

SEARCH_NUMB = 50


def func(x, y):
	res = np.array((2, 1))
	
	res[0] = y[1]
	res[1] = -DEF_P * x * cos(y[0])

	return res


def runge_kutta_step(x, y, h):
	k1 = func(x, y)
	k2 = func(x + h/2, y + h/2 * k1)
	k3 = func(x + h/2, y + h/2 * k2)
	k4 = func(x + h,   y + h * k3)

	res = y + h/6 * (k1 + 2 * k2 + 2 * k3 + k4)

	return res


def runge_kutta(y0_init, y1_init):
	oldRes = np.array([y0_init, y1_init])
	newRes = np.array([0.0, 0.0])
	x = 0

	listX = []
	listY0 = []
	listY1 = []

	listX.append(x)
	listY0.append(oldRes[0])
	listY1.append(oldRes[1])

	x += STEP_DEF

	for i in range(0, ITER_NUMB_DEF):
		newRes = runge_kutta_step(x, oldRes, STEP_DEF)

		oldRes = newRes
		x += STEP_DEF

		listX.append(x)
		listY0.append(newRes[0])
		listY1.append(newRes[1])



	return (listX, listY0)


def shooting(left, right, nparts):
	for i in range(SEARCH_NUMB):
		res = runge_kutta((right + left)/2, Y1_DEF)
		y_res = res[1][-1]

		if(y_res > Y_LEFT):
			right = (left + right) / 2
		else:
			left  = (left + right) / 2
            
	return res[1][0]



y_res = shooting(-1, 2, 1000)

print(y_res)

#res2 = runge_kutta(y_res, Y1_DEF)

#plt.plot(res2[0], res2[1], label = "shooting");

#plt.legend()
#plt.show()
