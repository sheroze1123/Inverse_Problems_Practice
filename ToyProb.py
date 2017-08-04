from fenics import *
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
 
u_D = Expression('x[0] == 0 && x[1] <= 1 ? 1 : 0.00001', degree=0)
 
def boundary(x, on_boundary):
    return on_boundary
 
bc = DirichletBC(V, u_D, boundary)
 
gamma = Constant(0.02)
sigma = Expression('x[0] >= 0.5 && x[0] <= 0.7 && x[1] >= 0.5 && x[1] <= 0.7 ? 0.2 : 0.1', degree=0)
u = TrialFunction(V)
v = TestFunction(V)
a = (gamma*dot(grad(u), grad(v)) + sigma*u*v)*dx
f = Constant(0)
L = f*v*dx
 
u = Function(V)
solve(a==L, u, bc)
 
# Save solution to file in VTK format
vtkfile = File('DiffusionEqFEniCS.pvd')
vtkfile << u
 
# Compute error in L2 norm
error_L2 = errornorm(u_D, u, 'L2')
 
# Compute maximum error at vertices
vertex_values_u_D = u_D.compute_vertex_values(mesh)
vertex_values_u = u.compute_vertex_values(mesh)
import numpy as np
error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))
 
# Print errors
print('error_L2  =', error_L2)
print('error_max =', error_max)

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

sigma_star_f = H_star/u_star
sigma_star = project(sigma_star_f, V)


# Plotting of sigma*
fig = plt.figure()
ax = fig.gca(projection='3d')

sigma_star_box = structured_mesh(sigma_star, (cells_per_side, cells_per_side))
sigma_star_ = sigma_star_box.values
cv = sigma_star_box.grid.coorv

ax.plot_surface(cv[0], cv[1], sigma_star_, cmap=cm.coolwarm, rstride=1, cstride=1)

plt.show()

interactive()
