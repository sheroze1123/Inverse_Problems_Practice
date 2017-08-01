from fenics import *
 
cells_per_side = 32
mesh = UnitSquareMesh(cells_per_side,cells_per_side)
V = FunctionSpace(mesh, 'P', 1)
 
u_D = Expression('x[0] == 0 && x[1] <= 1 ? 1 : 0.00001')
 
def boundary(x, on_boundary):
    return on_boundary
 
bc = DirichletBC(V, u_D, boundary)
 
gamma = Constant(0.02)
sigma = Expression('x[0] >= 0.5 && x[0] <= 0.7 && x[1] >= 0.5 && x[1] <= 0.7 ? 0.2 : 0.1')
u = TrialFunction(V)
v = TestFunction(V)
a = (gamma*dot(grad(u), grad(v)) + sigma*u*v)*dx
f = Constant(0)
L = f*v*dx
 
u = Function(V)
solve(a==L, u, bc)
# Plot solution and mesh
 
plot(u)
plot(mesh)
 
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
plot(H)

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

#  sigma_star = H_star/u_star
#  plot(sigma_star)
sigma_star = H_star/u_star
plot(sigma_star)

interactive()
