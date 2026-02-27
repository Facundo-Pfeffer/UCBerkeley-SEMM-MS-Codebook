# Example from Lecture January 28, 2026
import numpy as np
import matplotlib.pyplot as plt

# Properties
k = 1000
Q = 10
qn= -25
L = 10

# Number of terms in expansion
N = 2
K = np.zeros((N,N))
F = np.zeros(N)

# Assemble approximation matrices
for i in range(N):
  I = i + 1
  F[i] = L*Q*(1 - 1/2**(I+1))/(I+1) + qn
  for j in range(N):
    J = j + 1
    K[i,j] = (k/L)*I*J/(I+J-1)

# Compute coefficients of the approximation
ua = np.linalg.solve(K,F)

# Macauley bracket
def mcb(x):
  return np.where(x < 0, 0.0, x)

# Exact solution
x  = np.linspace(0,L,200)
ue = (-0.5*Q*mcb(x-L/2)**2 + ((L/2)*Q+qn)*x)/k
q  = -k*(-Q*mcb(x-L/2)+((L/2)*Q+qn))/k

# Evaluate approximate solution
uhat = np.zeros_like(x)
qhat = np.zeros_like(x)
for i in range(N):
  I = i + 1
  uhat = uhat + ua[i] * (x/L)**I
  qhat = qhat - k*ua[i] * I * (x/L)**(I-1) / L

# Plot results
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,5))

ax1.plot(x,ue)         # Exact temperature
ax1.plot(x,uhat)       # Approximate temperature
ax1.set_ylabel('u(x)')

ax2.plot(x,q)          # Exact flux
ax2.plot(x,qhat)       # Approximate flux
ax2.set_ylabel('q(x)')

plt.show()