You have one new message.

Skip to content
Using UC Berkeley Mail with screen readers
miguel 

Conversations
Tareas CE222

Facundo Leguizamon Pfeffer <facundo.pfeffer@berkeley.edu>
Attachments
Feb 6, 2026, 12:56 PM (6 days ago)
to Miguel

Hola Miguel!

Como me pediste, te adjunto los enunciados de las tareas. Por el momento vienen basándose únicamente en el libro, pero te voy a seguir adjuntando conforme avancemos con el curso.

Un saludo y nos vemos pronto!
Facundo
 2 Attachments
  •  Scanned by Gmail

Facundo Leguizamon Pfeffer <facundo.pfeffer@berkeley.edu>
Attachments
Feb 9, 2026, 9:45 PM (3 days ago)
to Miguel

Hola Miguel,

Te adjunto el enunciado de la última tarea. Primer tarea oficial de programación! 

Un saludo,
Facundo
 2 Attachments
  •  Scanned by Gmail
miguel.gomez.f@berkeley.edu. Press tab to insert.
import numpy as np
import matplotlib.pyplot as plt

# Define variables of the problem, domain and other paramters as needed
L  = # length of domain
u0 = # boundary values
uL = # boundary values
def b(x):
  return  # distributed loading function

# Create datastructures to define the mesh
N  =  # number of elements
x  =  # Nodal coordinates
lm =  # location matrix

# Define arrays to specify essential boundary conditions and the values
bc_ess = [0,N]     # Zero based indexing, node numbers for ess bcs 
bc_val = [u0,uL]   # The values at the ess nodes

# Generate exact solution for plotting
xtrue = np.linspace(0,L,200)
utrue = 

# Loop over elements to construct K matrix 
K = np.zeros((N+1,N+1))
for i in np.arange(N):
  klocal =           # Compute local stiffness, hand comp for linear elem
  rows   =           # Array of the global rows associated with klocal
  cols   = rows
  # Assemble local into global
  K[np.ix_(rows,cols)] = K[np.ix_(rows,cols)] + 

# Loop over elements, contstruct load vector F using Gauss Quadrature
F = np.zeros(N+1)
for i in np.arange(N):
  flocal =           # set up flocal using quadrature (2x1 array)
  rows   =           # Array of the global rows associated with flocal
  F[rows] = F[rows] + flocal


# Solve the equations accounting for the essential boundary conditions
u = np.zeros(N+1)

u[bc_ess] = # Set essential boundary values

## From here the code does not need to be changed unless you
## want to make it fancier

# Create mask for active degrees of freedom
mask         = np.ones(N+1, dtype=bool)
mask[bc_ess] = False

u[mask] = np.linalg.solve( K[np.ix_(mask,mask)], 
                           F[mask]-K[np.ix_(mask,bc_ess)]@bc_val)

# Plot FEA and exact solutions
plt.plot(x,u,'s-',label='FEA')
plt.plot(xtrue,utrue,label='exact')
plt.legend()
plt.show()
