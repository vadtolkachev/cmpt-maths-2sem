import numpy as np
from numpy import linalg
import math
from matplotlib import pylab as plt
from mpl_toolkits.mplot3d import Axes3D


DEF_K = 0

DEF_A = 0.5 + 0.1 * DEF_K


def graph3D(matrix, **kwargs):
	(x, y) = np.meshgrid(np.arange(matrix.shape[1]), np.arange(matrix.shape[0]))
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	surf = ax.plot_surface(x, y, matrix, **kwargs)
	return (fig, ax, surf)


def solve(dx, dt, step_x, step_t):
	v =  np.empty(shape=[step_t+1,step_x+1], dtype=float)
	for i in range(step_x+1):
		v[0][i] = 0

	u =  np.empty(shape=[step_t+1,step_x+1], dtype=float)
	for i in range(step_x+1):
		u[0][i] = 1 / (i*dx + DEF_A)
	for j in range(1,step_t):
		u[j][0] = 1 / (j*dt + DEF_A)
		u[j][step_x] = 1 / (j*dt + DEF_A + 1)
	    
	t_arr = []
	x_arr = []

	for i in range(0, step_t+1):
		t_arr.append( float(i*dt) )
	for i in range(0, step_x+1):
		x_arr.append( float(i*dx) )


	for i in range(1, step_t):
		for j in range(1, step_x):
			u[i+1][j] = u[i][j] + dt*( (v[i][j+1]-v[i][j-1])/(2*dx)+(dt*(u[i][j+1]-2*u[i][j]+u[i][j-1]))/(2*dx**2))
			v[i+1][j] = v[i][j] + dt*( (u[i][j+1]-u[i][j-1])/(2*dx)+(dt*(v[i][j+1]-2*v[i][j]+v[i][j-1]))/(2*dx**2))


	(fig, ax, surf) = graph3D(u, cmap=plt.cm.coolwarm)

	fig.colorbar(surf)

	ax.set_xlabel(' dx ')
	ax.set_ylabel(' dt ')
	ax.set_zlabel(' u ')

	plt.show()

	return u	


dt = 10**-3
dx = 2*10**-3
step_x = int(1/dx)
step_t = int(1/dt)


u = solve(dx, dt, step_x, step_t)


x = []
x_arr = []

for i in range(0, step_x+1):
	x_arr.append( float(i*dx) )
for i in range(0,step_x+1):
	x.append(float(u[800][i]))


plt.plot(x_arr, x)
plt.show()