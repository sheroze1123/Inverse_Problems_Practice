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

cells_per_side = 32
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

H_stars = []

q_degree = 3
dx = dx(metadata={'quadrature_degree': q_degree})

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
 
    # Ground truth data
    H = sigma * u

    # Adding perturbations to the ground truth to begin reconstruction
    rand_max = 0.05
    rand_min = -0.05
    rand_field = Function(V)
    rand_field_data = (np.random.rand(V.dim()) * 0.1) - 0.05
    rand_field.vector().set_local(rand_field_data)
    H_star = H * (1 + rand_field)

    H_stars.append(H_star) # Do I need the deepcopy here?


####################################################################
## Gradient descent
####################################################################

converged = False
maxiter = 1000
iterations = 0

# Random initialization of gamma and sigma reconstructions
gamma_recons = Function(V)
gamma_recons.vector().set_local(np.random.rand(V.dim()))
sigma_recons = Function(V)
sigma_recons.vector().set_local(np.random.rand(V.dim()))

gamma_learn_rate = 0.12
sigma_learn_rate = 0.12

# Run until desired convergence is obtained or until max iterations
while not (converged or iterations == maxiter):
    iterations += 1

    dPhi_dGamma = Constant(0)
    dPhi_dSigma = Constant(0)

    u_js = []
    w_js = []
    for j in range(0,len(u_Ds)):
        u_D = u_Ds[j]
        bc = DirichletBC(V, u_D, boundary)
        u_j = TrialFunction(V)
        v = TestFunction(V)
        a_j = (gamma_recons*dot(grad(u_j), grad(v)) + sigma_recons*u_j*v)*dx
        f = Constant(0)
        L = f*v*dx
        u_j = Function(V)
        solve(a_j==L, u_j, bc)
        u_js.append(u_j)
    

        # Solve the adjoint equation
        bc = DirichletBC(V, Constant(0), boundary)
        w_j = TrialFunction(V)
        v = TestFunction(V)
        a_w_j = (gamma_recons*dot(grad(w_j), grad(v)) + sigma_recons*w_j*v)*dx
        f = (sigma_recons*u_j - H_stars[j]) * sigma_recons
        L = f * v * dx
        w_j = Function(V)
        solve(a_w_j==L, w_j, bc)
        w_js.append(w_j)

        # TODO: Add regularization here
        dPhi_dGamma = dPhi_dGamma + div(grad(u_j))*w_j
        dPhi_dSigma = dPhi_dSigma + ((sigma_recons * u_j - H_stars[j])*u_j - w_j*u_j)

    gamma_recons = gamma_recons - dPhi_dGamma * gamma_learn_rate
    sigma_recons = sigma_recons - dPhi_dSigma * sigma_learn_rate

    sigma_error = np.abs(np.sum(project(sigma_recons - sigma, V).vector().array()))
    gamma_error = np.abs(np.sum(project(gamma_recons - gamma, V).vector().array()))

    print ('Sigma error: {:4.2f}, Gamma error: {:4.2f}'.format(sigma_error, gamma_error))
    if sigma_error < 3:
        converged = True



# Plotting of sigma_recons
fig = plt.figure()
ax = fig.gca(projection='3d')



sigma_box = structured_mesh(project(sigma_recons, V), (cells_per_side, cells_per_side))
sigma_ = sigma_box.values
cv = sigma_box.grid.coorv

ax.plot_surface(cv[0], cv[1], sigma_, cmap=cm.coolwarm, rstride=1, cstride=1)

plt.show()
# plot(sigma_star_avg)
# plot(sigma_stars[0])
interactive()
