from fenics import *
import numpy as np
 
cells_per_side = 64
mesh = UnitSquareMesh(cells_per_side,cells_per_side)
V = FunctionSpace(mesh, 'P', 1)
 
u_D_0 = Expression('x[0] == 0 && x[1] <= 1 ? 1 : 0.00001')
u_D_1 = Expression('x[0] == 0 && x[1] <= 1 ? sin(10*x[1]/pi) : 0.00001')
u_D_2 = Expression('x[0] == 1 && x[1] <= 1 ? 1 : 0.00001')
u_D_3 = Expression('x[1] == 0 && x[0] <= 1 ? 1 : 0.00001')
u_D_4 = Expression('x[1] == 1 && x[0] <= 1 ? 1 : 0.00001')
u_D_5 = Expression('x[0] == 1 && x[1] <= 1 ? 3 : 1.00001')
u_D_6 = Expression('x[1] == 0 && x[0] <= 1 ? 2 : 0.50001')
u_D_7 = Expression('x[1] == 1 && x[0] <= 1 ? 1 : 0.20001')
u_Ds = [u_D_0, u_D_1, u_D_2, u_D_3, u_D_4, u_D_5, u_D_6, u_D_7]
 
def boundary(x, on_boundary):
    return on_boundary
 
gamma = Constant(0.02)
sigma = Expression('x[0] >= 0.5 && x[0] <= 0.7 && x[1] >= 0.5 && x[1] <= 0.7 ? 0.2 : 0.1')
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
    sigma_stars.append(sigma_star)

sigma_star_avg = Constant(0)
for sigma_star in sigma_stars:
    sigma_star_avg = sigma_star + sigma_star_avg

sigma_star_avg = sigma_star_avg / len(sigma_stars)
plot(sigma_star_avg)
plot(sigma_stars[0])
interactive()
