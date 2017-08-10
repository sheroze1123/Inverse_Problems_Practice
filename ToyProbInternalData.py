from fenics import *
import numpy as np
parameters['plotting_backend'] == 'matplotlib'
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import axes3d
 
def structured_mesh(u, divisions):
    """
    Represent u on a structured mesh.
    """
    # u must have P1 elements, otherwise interpolate to P1 elements
    u2 = u if u.ufl_element().degree() == 1 else interpolate(u, FunctionSpace(mesh, 'P', 1))
    mesh = u.function_space().mesh()
    from BoxField import fenics_function2BoxField
    u_box = fenics_function2BoxField(u2, mesh, divisions, uniform_mesh=True)
    return u_box

cells_per_side = 128
mesh = UnitSquareMesh(cells_per_side,cells_per_side)
V = FunctionSpace(mesh, 'P', 1)
 
P1 = FiniteElement("Lagrange", triangle, 1)
u_D_0 = Expression('x[0] == 0 && x[1] <= 1 ? 1 : 0.00001', degree=0)
u_D_1 = Expression('x[0] == 0 && x[1] <= 1 ? sin(10*x[1]/pi) : 0.00001', element=P1)
u_D_2 = Expression('x[0] == 1 && x[1] <= 1 ? 1 : 0.00001', degree=0)
u_D_3 = Expression('x[1] == 0 && x[0] <= 1 ? 1 : 0.00001', degree=0)
u_D_4 = Expression('x[1] == 1 && x[0] <= 1 ? 1 : 0.00001', degree=0)
u_D_5 = Expression('x[0] == 1 && x[1] <= 1 ? 3 : 1.00001', degree=0)
u_D_6 = Expression('x[1] == 0 && x[0] <= 1 ? 2 : 0.50001', degree=0)
u_D_7 = Expression('x[1] == 1 && x[0] <= 1 ? 1 : 0.20001', degree=0)
u_Ds = [u_D_0, u_D_1, u_D_2, u_D_3, u_D_4, u_D_5, u_D_6, u_D_7]
 
def boundary(x, on_boundary):
    return on_boundary
 
gamma = Expression('x[0] >= 0.4 && x[0] <= 0.6 && x[1] >= 0.4 && x[1] <= 0.6 ? 0.03 : 0.01', degree=0)
sigma = Expression('x[0] >= 0.2 && x[0] <= 0.8 && x[1] >= 0.2 && x[1] <= 0.3 ? 0.2 : 0.1', degree=0)
# sigma = Expression('exp(-0.5 * (pow((x[0] - 0.6)/0.2, 2) + pow((x[1] - 0.7)/0.2, 2)))', degree=1)
sigma_stars = []

for u_D in u_Ds:
    bc = DirichletBC(V, u_D, boundary)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = (gamma*dot(grad(u), grad(v)) + sigma*u*v)*dx
    f = Constant(0)
    L = f*v*dx
     
    u = Function(V)
    solve(a==L, u, bc)
    # Plot solution and mesh
     
    #  plot(u)
    #  plot(mesh)
 
    H = sigma * u

    rand_max = 0.05
    rand_min = -0.05
    rand_field = Function(V)
    rand_field_data = (np.random.rand(V.dim()) * 0.1) - 0.05
    rand_field.vector().set_local(rand_field_data)
    H_star = H * (1 + rand_field)

    u_star = TrialFunction(V)
    a_star = (gamma*dot(grad(u_star), grad(v)) + H_star)*dx
    L = f*v*dx

    u_star = Function(V)
    solve(a==L, u_star, bc)

    sigma_star = H_star/u_star
    sigma_stars.append(project(sigma_star, V))

sigma_star_avg = Constant(0)
for sigma_star in sigma_stars:
    sigma_star_avg = sigma_star + sigma_star_avg

sigma_star_avg = sigma_star_avg / len(sigma_stars)
sigma_star_avg = project(sigma_star_avg, V)

# Plotting of sigma*
fig = plt.figure()
ax = fig.gca(projection='3d')

sigma_star_box = structured_mesh(sigma_star_avg, (cells_per_side, cells_per_side))
sigma_star_ = sigma_star_box.values
cv = sigma_star_box.grid.coorv

ax.plot_surface(cv[0], cv[1], sigma_star_, cmap=cm.coolwarm, rstride=1, cstride=1)

plt.show()
# plot(sigma_star_avg)
# plot(sigma_stars[0])
interactive()
