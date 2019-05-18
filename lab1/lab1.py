import matplotlib.pylab as plt
from math import exp
import numpy as np

EXP_NUMB = 100000

EXP_STEP = 0.0001

EXP_A = 998 * EXP_STEP + 1
EXP_B = 3996 * EXP_STEP
EXP_C = -499.5 * EXP_STEP
EXP_D = -1999 * EXP_STEP + 1

IMP_NUMB = 30

IMP_STEP = 0.0001

IMP_A_S = 998 * IMP_STEP - 1
IMP_B_S = 3996 * IMP_STEP
IMP_C_S = -449.5 * IMP_STEP
IMP_D_S = -1999 * IMP_STEP - 1

IMP_DENOM = IMP_A_S * IMP_D_S - IMP_C_S * IMP_B_S

IMP_A = -IMP_D_S / IMP_DENOM
IMP_B = IMP_B_S / IMP_DENOM
IMP_C = IMP_C_S / IMP_DENOM
IMP_D = -IMP_A_S / IMP_DENOM



def exact_solution():
	res = np.array([0.0, 0.0])
	list1 = []
	list2 = []

	i = 0
	t = 0.0
	while(t <= 10.0):
		res[0] = -5*exp(-1000*t) + 6*exp(-t)
		res[1] = 2.5*exp(-1000*t) - 1.5*exp(-t)
		
		list1.append(res[0])
		list2.append(res[1])

		t += 0.1
		i += 1


	plt.plot(list1, list2, label="exact_solution");


def explicid_euler_method():
	oldRes = np.array([1.0, 1.0])
	newRes = np.array([0.0, 0.0])

	list1 = []
	list2 = []

	list1.append(oldRes[0])
	list2.append(oldRes[1])

	for i in range(0, EXP_NUMB):
		newRes[0] = EXP_A * oldRes[0] + EXP_B * oldRes[1]
		newRes[1] = EXP_C * oldRes[0] + EXP_D * oldRes[1]

		oldRes = newRes

		list1.append(newRes[0])
		list2.append(newRes[1])

	plt.plot(list1, list2, label="explicid_euler_method");


def implicid_euler_method():
	oldRes = np.array([1.0, 1.0])
	newRes = np.array([0.0, 0.0])

	list1 = []
	list2 = []

	list1.append(oldRes[0])
	list2.append(oldRes[1])

	for i in range(0, IMP_NUMB):
		newRes[0] = IMP_A * oldRes[0] + IMP_B * oldRes[1]
		newRes[1] = IMP_C * oldRes[0] + IMP_D * oldRes[1]

		oldRes = newRes

		list1.append(newRes[0])
		list2.append(newRes[1])

	plt.plot(list1, list2, label="implicid_euler_method");



exact_solution()
explicid_euler_method()
implicid_euler_method()

plt.legend();
plt.show();
