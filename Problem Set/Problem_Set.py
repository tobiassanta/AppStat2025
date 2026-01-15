#!/usr/bin/env python
# coding: utf-8

# In[67]:


import matplotlib.pyplot as plt
import numpy as np
import scienceplots
plt.style.use(['science','notebook','grid'])
import sympy as sp
import random
from IPython.display import display, Math
import pandas as pd
from scipy.special import erfc 
from IPython.core.display import Latex
from sympy import *
import iminuit


def lprint(*args,**kwargs):
    """Pretty print arguments as LaTeX using IPython display system 

    Parameters
    ----------
    args : tuple 
        What to print (in LaTeX math mode)
    kwargs : dict 
        optional keywords to pass to `display` 
    """
    display(Latex('$$'+' '.join(args)+'$$'),**kwargs)


# # Problem 1

# ### Problem 1.1

# In[15]:


def simulate_pushup_days(num_pushup_days):
    waiting_times = []
    pushups_per_day = []

    days = 0
    while len(waiting_times) < num_pushup_days:
        roll = random.randint(1,6)
        if roll == 6:
            waiting_times.append(days)

            pushups = sum(1 for _ in range(120) if random.randint(1,6) == 6)
            pushups_per_day.append(pushups)

            days = 0
        else:
            days += 1

    return np.array(waiting_times), np.array(pushups_per_day)


# In[16]:


#Distribution of days between push-ups

waiting_times, pushups_per_day = simulate_pushup_days(100000)

plt.figure(figsize = (16,6))
d_waiting_times                       =   np.diff(np.unique(waiting_times)).min()
d_pushups_per_day                     =   np.diff(np.unique(pushups_per_day)).min()
left_of_first_bin_waiting_times       =   waiting_times.min() - float(d_waiting_times)/2
right_of_first_bin_waiting_times      =   waiting_times.max() + float(d_waiting_times)/2
left_of_first_bin_pushups_per_day     =   pushups_per_day.min() - float(d_pushups_per_day)/2
right_of_first_bin_pushups_per_day    =   pushups_per_day.max() + float(d_pushups_per_day)/2

plt.subplot(1,2,1)
plt.hist(waiting_times, bins = np.arange(left_of_first_bin_waiting_times, right_of_first_bin_waiting_times + d_waiting_times, d_waiting_times), density=True,
         edgecolor = 'white', color = 'black')
plt.title("Distribution of Days Between Push-ups")
plt.xlabel("Days between push-ups")
plt.ylabel("Probability")
plt.grid(axis = 'y', linestyle = '--', alpha = 0.5)

plt.subplot(1,2,2)
plt.hist(pushups_per_day, bins = np.arange(left_of_first_bin_pushups_per_day, right_of_first_bin_pushups_per_day + d_pushups_per_day, d_pushups_per_day), density=True, 
         edgecolor = 'white', color = 'black')
plt.title("Distribution of Push-Ups per Push-up Day")
plt.xlabel("Number of Push-ups")
plt.ylabel("Probability")
plt.grid(axis = 'y', linestyle = '--', alpha = 0.5)

plt.tight_layout()
plt.savefig('Problem 1.1.1.png', dpi = 300)


# In[17]:


#Mean, median, and standard deviation of number of push-ups in 10 days
def simulate_pushups(num_days):
    pushups_per_day = []

    for _ in range(num_days):
        roll = random.randint(1,6)

        if roll == 6:
            pushups = sum(1 for _ in range(120) if random.randint(1,6) == 6)

        else:
            pushups = 0

        pushups_per_day.append(pushups)

    return np.array(pushups_per_day)


# In[18]:


#Simulation
N_experiments = 150000
totals = []

for _ in range(N_experiments):
    pushups = simulate_pushups(10)
    totals.append(pushups.sum())

totals = np.array(totals)

display(Math(fr"\mu = {np.mean(totals):.2f}"))
display(Math(fr"Median = {np.median(totals):.2f}"))
display(Math(fr"\sigma =  {np.std(totals):.2f}"))


# In[19]:


plt.figure(figsize=(8,6))

d_totals = np.diff(np.unique(totals)).min()
left_of_first_bin_totals = totals.min() - d_totals/2
right_of_first_bin_totals = totals.max() + d_totals/2

plt.subplot(1,1,1)
plt.hist(totals, bins=np.arange(left_of_first_bin_totals, right_of_first_bin_totals + d_totals, d_totals), density=True, edgecolor='white', color='black')
plt.xlim(-5, 110)
plt.title("Distribution of Push-Ups in 10 Days")
plt.xlabel("Number of Push-Ups")
plt.ylabel("Probability")

mu = np.mean(totals)
med = np.median(totals)
sigma = np.std(totals)
entries = len(totals)

plot_info = [
    f"Entries: {entries}",
    f"Mean: {mu:.1f}",
    f"Median: {med:.1f}",
    f"Std: {sigma:.1f}",
]

plt.text(
    0.57, 0.93,
    "\n".join(plot_info),
    transform=plt.gca().transAxes,
    fontsize=12,
    family='monospace',
    verticalalignment='top',
    bbox=dict(
        facecolor='white',      
        edgecolor='black',      
        boxstyle='round,pad=0.3', 
        alpha=0.9               
    )
)

plt.tight_layout()
plt.savefig('Problem 1.1.2.png', dpi = 300)


# # Problem 2

# ### Problem 2.1

# In[20]:


mu_r_earth   =  149*10e-06
sig_r_earth  =  3*10e-06
mu_r_sun     =  25*10e-06
sig_r_sun    =  5*10e-06
mu_r_comet   =  309*10e-06
sig_r_comet  =  20*10e-06


# In[21]:


# Define variables:
f, r_earth,r_sun,r_comet = symbols("f, r_e, r_s, r_C")
df,dr_earth,dr_sun,dr_comet = symbols("sigma_f, sigma_e, sigma_s, sigma_c")

f = (r_earth - r_comet)/(r_sun - r_comet)
lprint(latex(Eq(symbols('f'),f)))

# Calculate uncertainty and print:
df = sqrt((f.diff(r_earth) * dr_earth)**2 + (f.diff(r_sun) * dr_sun)**2 + (f.diff(r_comet) * dr_comet)**2)
lprint(latex(Eq(symbols('sigma_f'), df)))

# Turn expression into numerical functions 
ff = lambdify((r_earth,r_sun, r_comet),f)
fdf = lambdify((r_earth,dr_earth,r_sun,dr_sun,r_comet, dr_comet),df)

# Numerically evaluate expressions and print 
vf = ff(mu_r_earth,mu_r_sun,mu_r_comet)
vdf = fdf(mu_r_earth,sig_r_earth,mu_r_sun,sig_r_sun,mu_r_comet,sig_r_comet)
lprint(fr'f = {vf:.2f} \pm {vdf:.2f}')


# ### Problem 2.2

# In[22]:


mu_N   =  1971
sig_N  =  np.sqrt(1971)
mu_t   =  98.4
sig_t  =  3.7


# In[23]:


# Define variables:
r, N, t = symbols("r, N, t")
dr, dN, dt = symbols("sigma_r, sigma_N, sigma_t")

r = N/t
lprint(latex(Eq(symbols('r'),r)))

dr= sqrt((r.diff(N) * dN)**2 + (r.diff(t) * dt)**2)
lprint(latex(Eq(symbols('sigma_r'), dr)))

fr = lambdify((N, t),r, 'numpy')
fdr = lambdify((N, dN, t, dt),dr, 'numpy')

vr = fr(mu_N, mu_t)
vdr = fdr(mu_N, sig_N, mu_t, sig_t)
lprint(fr'r = {vr:.1f} \pm {vdr:.1f}')


# ### Problem 2.3

# ### Without uncertainties

# In[24]:


def chauvenet(y):
    mean = np.mean(y)
    std  = np.std(y)
    N = len(y)
    criterion = 1/(2*N)

    d = np.abs(y - mean) / std
    d /= np.sqrt(2)
    prob = erfc(d)

    return prob >= criterion


# In[25]:


pylon_data = pd.read_csv('Problem 2/data_PylonPositions.csv', delimiter=',', header = 0)

pylon_data_sorted = pylon_data.sort_values(['# Position']).reset_index(drop = True)
pylon_data_sorted


# In[26]:


#Plotting of raw data
plt.figure(figsize = (12,8))
plt.hist(pylon_data_sorted['# Position'], bins = 120, color = 'black')
plt.xlabel('x position')
plt.ylabel('Frequency')
plt.title('Pylon Positions')
plt.tight_layout()
plt.savefig('Problem 2.3 - raw data.png', dpi = 300)
print(len(pylon_data_sorted))


# In[27]:


# Doing chauvenet on clusters
x = np.array(pylon_data_sorted['# Position'])

cluster1 = x[(x > 30) & (x < 70)]
cluster2 = x[(x > 70) & (x < 110)]
cluster3 = x[(x > 110) & (x < 150)]
cluster4 = x[(x > 150) & (x < 190)]

cluster1_mean   =   np.mean(cluster1)
cluster2_mean   =   np.mean(cluster2)
cluster3_mean   =   np.mean(cluster3)
cluster4_mean   =   np.mean(cluster4)

cluster1_std   =   np.std(cluster1)
cluster2_std   =   np.std(cluster2)
cluster3_std   =   np.std(cluster3)
cluster4_std   =   np.std(cluster4)

print(cluster1_mean, cluster1_std/np.sqrt(len(cluster1)))
print(cluster2_mean, cluster2_std/np.sqrt(len(cluster2)))
print(cluster3_mean, cluster3_std/np.sqrt(len(cluster3)))
print(cluster4_mean, cluster4_std/np.sqrt(len(cluster4)))


# ### With uncertainties

# In[28]:


x = np.array(pylon_data_sorted['# Position'])
sx = np.array(pylon_data_sorted['  Uncertainty'])

mask1 = (x > 30) & (x < 70)
mask2 = (x > 70) & (x < 110)
mask3 = (x > 110) & (x < 150)
mask4 = (x > 150) & (x < 190)

x1, s1 = x[mask1], sx[mask1]
x2, s2 = x[mask2], sx[mask2]
x3, s3 = x[mask3], sx[mask3]
x4, s4 = x[mask4], sx[mask4]

def weighted_mean(x, sx):
    w = 1.0 / sx**2
    mu = np.sum(w * x) / np.sum(w)
    err_mu = np.sqrt(1.0 / np.sum(w))
    return mu, err_mu

m1, e1 = weighted_mean(x1, s1)
m2, e2 = weighted_mean(x2, s2)
m3, e3 = weighted_mean(x3, s3)
m4, e4 = weighted_mean(x4, s4)

print(m1, "+/-", e1)
print(m2, "+/-", e2)
print(m3, "+/-", e3)
print(m4, "+/-", e4)

def reduced_chi2(x, sx, mu):
    return np.sum(((x - mu)/sx)**2) / (len(x) - 1)

e1 *= np.sqrt(reduced_chi2(x1, s1, m1))

import numpy as np
from scipy.stats import chi2

def chi2_stats(x, sx, mu):
    chi2_val = np.sum(((x - mu) / sx)**2)
    ndof = len(x) - 1
    chi2_red = chi2_val / ndof
    p_val = chi2.sf(chi2_val, ndof)
    return chi2_val, chi2_red, ndof, p_val

clusters = [
    (x1, s1, m1, "Cluster 1"),
    (x2, s2, m2, "Cluster 2"),
    (x3, s3, m3, "Cluster 3"),
    (x4, s4, m4, "Cluster 4"),
]

for x, sx, mu, label in clusters:
    chi2_val, chi2_red, ndof, p_val = chi2_stats(x, sx, mu)
    print(f"{label}:")
    print(f"  chi^2 = {chi2_val:.2f}")
    print(f"  ndof = {ndof}")
    print(f"  chi^2_red = {chi2_red:.2f}")
    print(f"  p-value = {p_val:.3e}\n")



# In[29]:


import numpy as np
from scipy.stats import chi2

# Define function for weighted mean with scatter correction
def weighted_mean_with_scatter(x, sx):
    """
    x : array of measurements
    sx: array of per-point uncertainties
    Returns: mu, err_mu, chi2, chi2_red, ndof, p-value
    """
    w = 1.0 / sx**2
    mu = np.sum(w * x) / np.sum(w)
    chi2_val = np.sum(((x - mu)/sx)**2)
    ndof = len(x) - 1
    chi2_red = chi2_val / ndof
    # Weighted SEM
    err_mu = np.sqrt(1.0 / np.sum(w))
    # Scale uncertainty if chi2_red > 1
    if chi2_red > 1:
        err_mu *= np.sqrt(chi2_red)
    # p-value
    p_val = chi2.sf(chi2_val, ndof)
    return mu, err_mu, chi2_val, chi2_red, ndof, p_val

# Example clusters (replace with your actual arrays)
clusters = [
    (x1, s1, "Cluster 1"),
    (x2, s2, "Cluster 2"),
    (x3, s3, "Cluster 3"),
    (x4, s4, "Cluster 4"),
]

# Compute and print results for each cluster
for x, sx, label in clusters:
    mu, err_mu, chi2_val, chi2_red, ndof, p_val = weighted_mean_with_scatter(x, sx)
    print(f"{label}:")
    print(f"  Weighted mean = {mu:.6f} ± {err_mu:.6f}")
    print(f"  chi^2 = {chi2_val:.2f}")
    print(f"  ndof = {ndof}")
    print(f"  chi^2_red = {chi2_red:.2f}")
    print(f"  p-value = {p_val:.3e}\n")


# ### Do measurements match?

# 

# In[30]:


import numpy as np
from scipy.stats import chi2

# Group 1 (no uncertainties provided, so using ± values as SEM)
p1_means = np.array([50.1, 90.5, 131.3, 172.1])
p1_errs  = np.array([0.7, 0.8, 0.8, 0.5])

# Group 2 (with uncertainties)
p2_means = np.array([50.0, 90.7, 130.9, 171.9])
p2_errs  = np.array([0.8, 0.6, 1.0, 0.6])

# 1️⃣ Check if the two groups match
print("Comparison of the two groups:")
for i, (m1, e1, m2, e2) in enumerate(zip(p1_means, p1_errs, p2_means, p2_errs), 1):
    diff = m1 - m2
    err_diff = np.sqrt(e1**2 + e2**2)
    z_score = diff / err_diff
    print(f"Pylon {i}: Δ = {diff:.2f} ± {err_diff:.2f}, Z = {z_score:.2f}")

# 2️⃣ Combine the groups using weighted mean
def weighted_mean(mu1, err1, mu2, err2):
    w1 = 1.0 / err1**2
    w2 = 1.0 / err2**2
    mu_comb = (w1*mu1 + w2*mu2) / (w1 + w2)
    err_comb = np.sqrt(1 / (w1 + w2))
    return mu_comb, err_comb

combined_means = []
combined_errs = []

for m1, e1, m2, e2 in zip(p1_means, p1_errs, p2_means, p2_errs):
    mu, err = weighted_mean(m1, e1, m2, e2)
    combined_means.append(mu)
    combined_errs.append(err)

combined_means = np.array(combined_means)
combined_errs = np.array(combined_errs)

print("\nCombined pylon positions:")
for i, (mu, err) in enumerate(zip(combined_means, combined_errs), 1):
    print(f"Pylon {i}: {mu:.2f} ± {err:.2f}")

# 3️⃣ Check if pylons are equidistant
dx = np.diff(combined_means)
dx_mean = np.mean(dx)
deviation = dx - dx_mean
dx_err = np.sqrt(combined_errs[:-1]**2 + combined_errs[1:]**2)
z_scores = deviation / dx_err

print("\nDistances between pylons:", dx)
print("Deviation from mean spacing:", deviation)
print("Z-scores:", z_scores)

# Optional: chi-square test for equidistant spacing
chi2_spacing = np.sum((deviation / dx_err)**2)
ndof = len(dx) - 1
p_val_spacing = chi2.sf(chi2_spacing, ndof)
print(f"\nChi2 spacing = {chi2_spacing:.2f}, ndof = {ndof}, p-value = {p_val_spacing:.3e}")


# # Problem 3 

# ### Problem 3.1

# ### Circles

# In[31]:


N_exp      = 100
N          = 100000

#Circle parameters
Ax, Ay, RA = 0, 0, 6
Bx, By, RB = 3, 7, 4

#List for multiple experiments
fraction_A_in_B_list = np.array([])
fraction_B_in_A_list = np.array([])

for i in range(N_exp):
    #Generation of points for circle A
    rA          = RA * np.sqrt(np.random.rand(N))
    theta_A     = 2 * np.pi * np.random.rand(N)
    xA          = Ax + rA * np.cos(theta_A)
    yA          = Ay + rA * np.sin(theta_A)

    #Generation of points for circle B
    rB          = RB * np.sqrt(np.random.rand(N))
    theta_B     = 2 * np.pi * np.random.rand(N)
    xB          = Bx + rB * np.cos(theta_B)
    yB          = By + rB * np.sin(theta_B)

    #Fraction of A overlapping B
    inside_B                = (xA-Bx)**2 + (yA-By)**2 <= RB**2
    fraction_A_in_B         = np.sum(inside_B)/N
    fraction_A_in_B_list    = np.append(fraction_A_in_B_list, fraction_A_in_B)


    #Fraction of B overlapping A
    inside_A    = (xB-Ax)**2 + (yB-Ay)**2 <= RA**2
    fraction_B_in_A     = np.sum(inside_A)/N
    fraction_B_in_A_list    = np.append(fraction_B_in_A_list, fraction_B_in_A)


# Check which points are also inside
plt.figure(figsize = (9,8))
plt.scatter(xA[~inside_B], yA[~inside_B], s=1, color='red', label='Circle A')
plt.scatter(xA[inside_B], yA[inside_B], s=1, color='green', label='A and B')

# Points in B
plt.scatter(xB[inside_A], yB[inside_A], s=1, color='green')
plt.scatter(xB[~inside_A], yB[~inside_A], s=1, color='black', label='Circle B')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(markerscale=10)
plt.grid()
plt.tight_layout()
plt.savefig('Problem 3.1.1.png', dpi = 300)
plt.show()

print(fr'Fraction A in B = {np.mean(fraction_A_in_B_list):.6f} +- {np.std(fraction_A_in_B_list, ddof = 1)/np.sqrt((N_exp)):.6f}')
print(fr'Fraction B in A = {np.mean(fraction_B_in_A_list):.6f} +- {np.std(fraction_B_in_A_list, ddof = 1)/np.sqrt(N_exp):.6f}')


# ### Hyperballs

# In[32]:


import numpy as np

# Parameters
N_exp = 100       # number of repeated experiments
N = 100_000       # number of points per experiment

# Hyperball centers and radii
A = np.array([0.0, 0.0, 0.0, 0.0])
B = np.array([3.0, 7.0, -1.0, 2.0])
RA, RB = 6.0, 4.0

# Lists to store fractions from each experiment
fraction_A_in_B_list = []
fraction_B_in_A_list = []

for i in range(N_exp):
    # --- Generate points in hyperball A ---
    vA = np.random.normal(size=(N, 4))
    vA /= np.linalg.norm(vA, axis=1)[:, None]
    rA = RA * np.random.rand(N)**(1/4)
    points_A = A + vA * rA[:, None]

    # Fraction of A overlapping B
    inside_B = np.sum((points_A - B)**2, axis=1) <= RB**2
    fraction_A_in_B_list.append(np.mean(inside_B))

    # --- Generate points in hyperball B ---
    vB = np.random.normal(size=(N, 4))
    vB /= np.linalg.norm(vB, axis=1)[:, None]
    rB = RB * np.random.rand(N)**(1/4)
    points_B = B + vB * rB[:, None]

    # Fraction of B overlapping A
    inside_A = np.sum((points_B - A)**2, axis=1) <= RA**2
    fraction_B_in_A_list.append(np.mean(inside_A))

# Convert to numpy arrays
fraction_A_in_B_list = np.array(fraction_A_in_B_list)
fraction_B_in_A_list = np.array(fraction_B_in_A_list)

# Compute mean and standard error over experiments
mean_A_in_B = np.mean(fraction_A_in_B_list)
err_A_in_B = np.std(fraction_A_in_B_list, ddof=1) / np.sqrt(N_exp)

mean_B_in_A = np.mean(fraction_B_in_A_list)
err_B_in_A = np.std(fraction_B_in_A_list, ddof=1) / np.sqrt(N_exp)

print(f"Fraction of A overlapping B = {mean_A_in_B:.5f} ± {err_A_in_B:.5f}")
print(f"Fraction of B overlapping A = {mean_B_in_A:.5f} ± {err_B_in_A:.5f}")


# ### Problem 3.2

# ### Spherical coordinate r

# In[33]:


N = 100000

# Generation of points in xyz
r = np.random.rand(N)**(1/3)
theta = 2*np.pi*np.random.rand(N)       
phi = np.arccos(1 - 2*np.random.rand(N))
x = r * np.sin(phi) * np.cos(theta)
y = r * np.sin(phi) * np.sin(theta)
z = r * np.cos(phi)

r_check = np.sqrt(x**2 + y**2 + z**2)


plt.figure(figsize = (12,8))
plt.hist(r_check, bins=50, color='black', edgecolor='white')
plt.xlabel('r')
plt.ylabel('Frequency')
plt.title('Radial Distribution for Uniform Sphere')
plt.tight_layout()
plt.savefig('Problem 3.2.1.png', dpi = 300)


# ### Distributions of theta and phi

# In[34]:


mask        = (z > 0) & (r < 1)
x_sel       = x[mask]
y_sel       = y[mask]
z_sel       = z[mask]

theta_sel   = np.arctan2(y_sel, x_sel)
phi_sel     = np.arccos(z_sel / np.sqrt(x_sel**2 + y_sel**2 + z_sel**2))

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.hist(theta_sel, bins=50, color='black', edgecolor='white')
plt.xlabel('θ (radians)')
plt.ylabel('Frequency')
plt.title('θ distribution')

plt.subplot(1,2,2)
plt.hist(phi_sel, bins=50, color='black', edgecolor='white')
plt.xlabel('φ (radians)')
plt.ylabel('Frequency')
plt.title('φ distribution')

plt.tight_layout()
plt.savefig('Problem 3.2.2.png', dpi = 300)
plt.show()


# ### Problem 3.2.3

# In[39]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import fsolve

# Parameters
N = 20000
v0 = 1.0

# --- 2) Accept-Reject method ---
v_max = 10 * v0   # choose large enough
f_max = (2*v0/v0)**2 * np.exp(-2*v0/v0)  # peak at v = 2 v0

def f(v):
    return (v/v0)**2 * np.exp(-v/v0)

v_accept = []
while len(v_accept) < N:
    v_trial = np.random.rand() * v_max
    u = np.random.rand() * f_max
    if u < f(v_trial):
        v_accept.append(v_trial)
v_accept = np.array(v_accept)

# --- Plot ---
bins = 55
v_plot = np.linspace(0, v_max, 500)
pdf = (v_plot/v0)**2 * np.exp(-v_plot/v0)
pdf /= np.trapz(pdf, v_plot)  # normalize

plt.hist(v_accept, bins=bins, density=True, label='Accept-Reject method', color='black', edgecolor = 'white')
plt.plot(v_plot, pdf, 'r-', lw=2, label='Analytical')
plt.xlabel('v')
plt.ylabel('Probability density')
plt.title('Velocity distribution from Accept-Reject Method')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Problem 3.2.3.png', dpi = 300)
plt.show()


# In[58]:


# --- Problem 3.2.4: Radial material distribution m_p(r) ---

N = 10000
v0 = 100.0       # m/s
g  = 9.82        # m/s^2

# --- Sample theta (upper hemisphere, isotropic) ---
theta = np.arccos(1 - np.random.rand(N))   # p(theta) = sin(theta)

# --- Sample velocity using accept-reject ---
v_max = 10 * v0
f_max = 4 * np.exp(-2)   # max of (v/v0)^2 exp(-v/v0)

def f(v):
    return (v / v0)**2 * np.exp(-v / v0)

v_vals = []
while len(v_vals) < N:
    v_trial = np.random.rand() * v_max
    u = np.random.rand() * f_max
    if u < f(v_trial):
        v_vals.append(v_trial)

v_vals = np.array(v_vals)

# --- Compute radial distance ---
r_vals = np.sin(theta) * v_vals**2 / g

# --- Analytic scaling for m_p(r) ---
r_plot = np.linspace(0, r_vals.max(), 500)

r0 = v0**2 / g
m_analytic = np.sqrt(r_plot) * np.exp(-np.sqrt(r_plot / r0))

# Normalize analytic curve
m_analytic /= np.trapz(m_analytic, r_plot)

# --- Plot ---
plt.figure(figsize=(8,6))
plt.hist(r_vals, bins=50, density=True,
         color='black', edgecolor='white', label='Simulation')

plt.xlabel('r (m)')
plt.ylabel(r'$m(r)$')
plt.title('Radial Material Distribution')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Problem 3.2.4.png', dpi=300)
plt.show()



# # Problem 4

# ### First simulation

# In[64]:


import numpy as np
import matplotlib.pyplot as plt

# Parameters
N_rolls = 200
N_dice  = 25

# -------------------------------
# Case 1: All fair dice
# -------------------------------
rolls_fair   = np.random.randint(1, 7, size=(N_rolls, N_dice))
results_fair = rolls_fair.flatten()

# -------------------------------
# Case 2: One fake die
# -------------------------------
fake_value = np.random.randint(1, 7)
print("Fake die value:", fake_value)

results_fake = []

for _ in range(N_rolls):
    fair_dice = np.random.randint(1, 7, size=N_dice - 1)
    box_roll  = np.append(fair_dice, fake_value)
    results_fake.extend(box_roll)

results_fake = np.array(results_fake)

# -------------------------------
# Common binning (same logic as push-up problem)
# -------------------------------
d_faces = np.diff(np.unique(results_fair)).min()

left_of_first_bin = results_fair.min() - d_faces / 2
right_of_last_bin = results_fair.max() + d_faces / 2

bins_faces = np.arange(left_of_first_bin,
                        right_of_last_bin + d_faces,
                        d_faces)

# -------------------------------
# Plot
# -------------------------------
plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
plt.hist(results_fair,
         bins=bins_faces,
         density=True,
         edgecolor='white',
         color='black')
plt.title('All Fair Dice')
plt.xlabel('Die face')
plt.ylabel('Probability')
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.subplot(1,2,2)
plt.hist(results_fake,
         bins=bins_faces,
         density=True,
         edgecolor='white',
         color='black')
plt.title('One Fake Die')
plt.xlabel('Die face')
plt.ylabel('Probability')
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('Problem_4.1.1_dice_two_cases.png', dpi=300)
plt.show()


# In[65]:


def chi_square_test(data):
    faces = np.arange(1, 7)
    observed = np.array([np.sum(data == f) for f in faces])
    expected = np.ones(6) * len(data) / 6

    chi2_stat = np.sum((observed - expected)**2 / expected)
    dof = 5
    p_value = 1 - chi2.cdf(chi2_stat, dof)

    return chi2_stat, p_value

chi2_fair, p_fair = chi_square_test(results_fair)
chi2_fake, p_fake = chi_square_test(results_fake)

print("All fair dice:")
print(f"  chi^2 = {chi2_fair:.2f}")
print(f"  p-value = {p_fair:.4f}")

print("\nOne fake die:")
print(f"  chi^2 = {chi2_fake:.2f}")
print(f"  p-value = {p_fake:.4e}")


# # Problem 5

# ### Plotting of data

# In[100]:


from scipy.stats import chisquare
# Load CSV
interspacing_data = pd.read_csv('Problem 5/data_InconstantBackground.csv', header=0)
distance = interspacing_data['# Distance']


# In[182]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare

# 1️⃣ Load data
interspacing_data = pd.read_csv('Problem 5/data_InconstantBackground.csv', header=0)
distance = interspacing_data['# Distance']

# -----------------------------
# 2️⃣ Full dataset histogram
# -----------------------------
xmin_all, xmax_all = distance.min(), distance.max()
Nbins_all = 1500  # you can adjust if needed #orignally 50

fig, ax = plt.subplots(figsize=(16,6))

# Classic histogram
hist = ax.hist(distance, bins=Nbins_all, range=(xmin_all, xmax_all),
               histtype='step', linewidth=1, color='black', label='Data')

# Counts, bin centers, Poisson errors
counts_all, bin_edges_all = np.histogram(distance, bins=Nbins_all, range=(xmin_all, xmax_all))
x_all = (bin_edges_all[1:] + bin_edges_all[:-1]) / 2
y_all = counts_all
sy_all = np.sqrt(counts_all)

# Add error bars
#ax.errorbar(x_all, y_all, yerr=sy_all, fmt='.k', ecolor='b', elinewidth=1, capsize=2, capthick=1, label='Poisson errors')

# Labels and title
ax.set(xlabel='Distance (nm)', ylabel='Counts', title='Histogram of All Molecular Interspacing Data')
ax.legend(loc='best')
#plt.savefig('Problem 5.1.2 1500bins.png', dpi = 300)
plt.show()

# -----------------------------
# 3️⃣ Background [8,10] nm histogram
# -----------------------------
background = distance[(distance >= 8) & (distance <= 10)]
xmin_bg, xmax_bg = 8, 10
Nbins_bg = 10  # 0.2 nm/bin #originally 10

fig, ax = plt.subplots(figsize=(10,5))

# Classic histogram
hist_bg = ax.hist(background, bins=Nbins_bg, range=(xmin_bg, xmax_bg),
                  histtype='step', linewidth=2, color='black', label='Data')

# Counts, bin centers, Poisson errors
counts_bg, bin_edges_bg = np.histogram(background, bins=Nbins_bg, range=(xmin_bg, xmax_bg))
x_bg = (bin_edges_bg[1:] + bin_edges_bg[:-1]) / 2
y_bg = counts_bg
sy_bg = np.sqrt(counts_bg)

# Add error bars
#ax.errorbar(x_bg, y_bg, yerr=sy_bg, fmt='.b', ecolor='blue', elinewidth=1, capsize=1, capthick=1, label='Poisson errors')

# Labels and title
ax.set(xlabel='Distance (nm)', ylabel='Counts', title='Histogram of Background [8,10] nm')
ax.legend(loc='best')
plt.tight_layout()
#plt.savefig('Problem 5.1.1.png', dpi = 300)
plt.show()

# -----------------------------
# 4️⃣ Chi-square test for uniformity in background
# -----------------------------
expected_bg = [len(background)/Nbins_bg] * Nbins_bg  # uniform expected counts
chi2_stat, p_value = chisquare(y_bg, f_exp=expected_bg)

print("Chi-square statistic:", chi2_stat)
print("p-value:", p_value)

if p_value > 0.05:
    print("Background [8,10] nm is consistent with a uniform distribution.")
else:
    print("Background [8,10] nm is NOT uniform.")


# ### Fitting

# In[183]:


import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from scipy.stats import norm


# In[281]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit

# -----------------------------
# 5️⃣ Specify peaks and fit/plot tails
# -----------------------------
peaks = [
    {'position': 0.9, 'fit_window': 0.3, 'plot_sigma_mult': 2.6,
     'color': 'red', 'alpha': 0.8, 'A_guess': 275, 'sigma_guess': 0.1, 'B_guess': 2},
    {'position': 3.4, 'fit_window': 0.4, 'plot_sigma_mult': 3,
     'color': 'blue', 'alpha': 0.8, 'A_guess': 150, 'sigma_guess': 0.05, 'B_guess': 10},
    {'position': 5.9, 'fit_window': 0.5, 'plot_sigma_mult': 3,
     'color': 'green', 'alpha': 0.8, 'A_guess': 100, 'sigma_guess': 0.05, 'B_guess': 8}
]

# -----------------------------
# 1️⃣ Load data
# -----------------------------
interspacing_data = pd.read_csv('Problem 5/data_InconstantBackground.csv', header=0)
distance = interspacing_data['# Distance']

# -----------------------------
# 2️⃣ Full dataset histogram
# -----------------------------
xmin_all, xmax_all = distance.min(), distance.max()
Nbins_all = 400  # high-resolution histogram #1500 before

fig, ax = plt.subplots(figsize=(16,6))

# Classic histogram (black step line)
hist = ax.hist(distance, bins=Nbins_all, range=(xmin_all, xmax_all),
               histtype='step', linewidth=1, color='black', label='Data')

# Counts, bin centers, Poisson errors
counts_all, bin_edges_all = np.histogram(distance, bins=Nbins_all, range=(xmin_all, xmax_all))
x_all = (bin_edges_all[1:] + bin_edges_all[:-1]) / 2
y_all = counts_all
sy_all = np.sqrt(counts_all)
sy_all[sy_all == 0] = 1  # avoid division by zero

# -----------------------------
# 3️⃣ Gaussian + background function
# -----------------------------
def gaussian(x, A, mu, sigma, B):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + B

# -----------------------------
# 4️⃣ Fit function for one peak
# -----------------------------
def fit_peak(x_data, y_data, y_err, peak_guess, fit_window=0.4):
    """
    Fit a single Gaussian + background for a peak.

    Parameters:
    - x_data, y_data, y_err: full histogram arrays
    - peak_guess: initial peak position
    - fit_window: half-width of the region used for fitting (tail for fit)
    """
    # Select local region for fitting
    mask = (x_data >= peak_guess - fit_window) & (x_data <= peak_guess + fit_window)
    x_peak = x_data[mask]
    y_peak = y_data[mask]
    sy_peak = y_err[mask]

    # Chi-square function for Minuit
    def chi2(A, mu, sigma, B):
        model = gaussian(x_peak, A, mu, sigma, B)
        return np.sum(((y_peak - model)/sy_peak)**2)

    # Initialize Minuit
    m = Minuit(chi2,
           A=p['A_guess'],
           mu=peak_guess,
           sigma=p['sigma_guess'],
           B=p['B_guess'])

    m.limits['A'] = (0, None)
    m.limits['sigma'] = (0, None)
    m.limits['B'] = (0, None)
    m.errordef = 1
    m.migrad()
    m.hesse()

    return m, x_peak, y_peak


fits = []

for p in peaks:
    m, x_peak, y_peak = fit_peak(x_all, y_all, sy_all,
                                 peak_guess=p['position'],
                                 fit_window=p['fit_window'])
    fits.append((m, x_peak, y_peak, p))  # save Minuit result and peak info

# -----------------------------
# 6️⃣ Overlay Gaussian fits with custom tails
# -----------------------------
for m, x_peak, y_peak, p in fits:
    mu = m.values['mu']
    sigma = m.values['sigma']
    A = m.values['A']
    B = m.values['B']

    # x values for plotting the Gaussian with long tail
    x_fit = np.linspace(mu - p['plot_sigma_mult']*sigma, mu + p['plot_sigma_mult']*sigma, 200)
    y_fit = gaussian(x_fit, A, mu, sigma, B)
    ax.plot(x_fit, y_fit, color=p['color'], linewidth=1.5, alpha = p['alpha'], label=f'Peak at {p["position"]} nm')
        # Text showing fitted parameters with uncertainties
    textstr = (f"A = {A:.1f} ± {m.errors['A']:.1f}\n"
               f"mu = {mu:.3f} ± {m.errors['mu']:.3f}\n"
               f"sigma = {sigma:.3f} ± {m.errors['sigma']:.3f}\n"
               f"B = {B:.1f} ± {m.errors['B']:.1f}")

    # Place the text just above the peak
    ax.text(mu + 0.3, max(y_fit) - 25, textstr, fontsize=10, color=p['color'],
            bbox=dict(facecolor='white', alpha=0.5))


# -----------------------------
# 7️⃣ Finalize plot
# -----------------------------
ax.set(xlabel='Distance (nm)', ylabel='Counts', title='Histogram with Individual Gaussian Fits')
ax.legend(loc='best')
plt.tight_layout()
#plt.savefig('Problem 5.1.2.png', dpi = 300)
plt.show()

for m, x_peak, y_peak, p in fits:
    A = m.values['A']
    sigma = m.values['sigma']
    err_A = m.errors['A']
    err_sigma = m.errors['sigma']

    intensity = A * sigma * np.sqrt(2*np.pi)
    intensity_err = np.sqrt((sigma*np.sqrt(2*np.pi)*err_A)**2 + (A*np.sqrt(2*np.pi)*err_sigma)**2)

    print(f"Peak at {p['position']} nm: intensity = {intensity:.1f} ± {intensity_err:.1f}")


# In[282]:


from scipy.stats import chi2
import numpy as np

# Compute intensities and errors
intensities = []
intensity_errors = []

for m, x_peak, y_peak, p in fits:
    A = m.values['A']
    sigma = m.values['sigma']
    err_A = m.errors['A']
    err_sigma = m.errors['sigma']

    I = A * sigma * np.sqrt(2*np.pi)
    err_I = np.sqrt((sigma*np.sqrt(2*np.pi)*err_A)**2 + (A*np.sqrt(2*np.pi)*err_sigma)**2)

    intensities.append(I)
    intensity_errors.append(err_I)

intensities = np.array(intensities)
intensity_errors = np.array(intensity_errors)

# Weighted average
weights = 1 / intensity_errors**2
I_bar = np.sum(intensities * weights) / np.sum(weights)

# Chi-square
chi2_val = np.sum((intensities - I_bar)**2 / intensity_errors**2)
dof = len(intensities) - 1
p_value = chi2.sf(chi2_val, dof)

print("Peak intensities:", intensities)
print("Intensity errors:", intensity_errors)
print(f"Weighted average intensity = {I_bar:.2f}")
print(f"Chi-square = {chi2_val:.2f} for {dof} dof")
print(f"P-value = {p_value:.3f}")


# # Problem 5.2

# ### East and West alignment

# In[289]:


import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Pyramid data: (Name, Year BC, East, East_unc, West, West_unc)
pyramids = [
    ("Meidum", 2600, -20.6, 1.0, -18.1, 1.0),
    ("Bent", 2583, -17.3, 0.2, -11.8, 0.2),
    ("Red", 2572, -8.7, 0.2, None, None),
    ("Khufu", 2554, -3.4, 0.2, -2.8, 0.2),
    ("Khafre", 2522, 6.0, 0.2, 6.0, 0.2),
    ("Menkaure", 2489, 12.4, 1.0, 14.1, 1.8),
    ("Sahure", 2446, 23, 10, None, None),
    ("Neferirkare", 2433, 30, 10, None, None),
]

# Lists for plotting and analysis
E_vals, W_vals, E_err, W_err, labels, years = [], [], [], [], [], []
delta_vals, delta_err, delta_years = [], [], []

print(f"{'Pyramid':<10} {'Δ (E-W)':>10} {'σΔ':>8} {'Agreement?':>15}")
print("-"*45)

for name, year, E, E_unc, W, W_unc in pyramids:
    if E is not None and W is not None:
        delta = E - W
        sigma_delta = math.sqrt(E_unc**2 + W_unc**2)
        significance = abs(delta) / sigma_delta
        if significance <= 1:
            agreement = "Good"
        elif significance <= 3:
            agreement = "Moderate"
        else:
            agreement = "Poor"
        print(f"{name:<10} {delta:10.2f} {sigma_delta:8.2f} {significance:.5f}")

        # Add to plotting and analysis lists
        E_vals.append(E)
        W_vals.append(W)
        E_err.append(E_unc)
        W_err.append(W_unc)
        labels.append(name)
        years.append(year)

        delta_vals.append(delta)
        delta_err.append(sigma_delta)
        delta_years.append(year)
    else:
        print(f"{name:<10} {'-':>10} {'-':>8} {'No data':>15}")


# In[290]:


import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Pyramid data: (Name, Year BC, East, East_unc, West, West_unc)
pyramids = [
    ("Meidum", 2600, -20.6, 1.0, -18.1, 1.0),
    ("Bent", 2583, -17.3, 0.2, -11.8, 0.2),
    ("Red", 2572, -8.7, 0.2, None, None),
    ("Khufu", 2554, -3.4, 0.2, -2.8, 0.2),
    ("Khafre", 2522, 6.0, 0.2, 6.0, 0.2),
    ("Menkaure", 2489, 12.4, 1.0, 14.1, 1.8),
    ("Sahure", 2446, 23, 10, None, None),
    ("Neferirkare", 2433, 30, 10, None, None),
]

print(
    f"{'Pyramid':<10} "
    f"{'Δ':>7} "
    f"{'σΔ':>7} "
    f"{'Δ/σΔ':>8} "
    f"{'σ_sys':>8} "
    f"{'σΔ_new':>9} "
    f"{'Δ/σΔ_new':>12}"
)
print("-" * 70)

for name, year, E, E_unc, W, W_unc in pyramids:
    if E is not None and W is not None:
        # Original difference and uncertainty
        delta = E - W
        sigma_delta = math.sqrt(E_unc**2 + W_unc**2)
        significance = abs(delta) / sigma_delta

        # Compute systematic uncertainty
        numerator = delta**2 - (E_unc**2 + W_unc**2)
        if numerator > 0:
            sigma_sys = math.sqrt(numerator / 2)
        else:
            sigma_sys = 0.0

        # Inflated uncertainty
        sigma_delta_new = math.sqrt(E_unc**2 + W_unc**2 + 2 * sigma_sys**2)
        significance_new = abs(delta) / sigma_delta_new

        print(
            f"{name:<10} "
            f"{delta:7.2f} "
            f"{sigma_delta:7.2f} "
            f"{significance:8.3f} "
            f"{sigma_sys:8.2f} "
            f"{sigma_delta_new:9.2f} "
            f"{significance_new:12.3f}"
        )
    else:
        print(
            f"{name:<10} "
            f"{'-':>7} "
            f"{'-':>7} "
            f"{'-':>8} "
            f"{'-':>8} "
            f"{'-':>9} "
            f"{'No data':>12}"
        )


# In[291]:


import math

# Pyramid data: (Name, East, East_unc, West, West_unc)
pyramids = [
    ("Meidum", -20.6, 1.0, -18.1, 1.0),
    ("Bent", -17.3, 0.2, -11.8, 0.2),
    ("Red", -8.7, 0.2, None, None),
    ("Khufu", -3.4, 0.2, -2.8, 0.2),
    ("Khafre", 6.0, 0.2, 6.0, 0.2),
    ("Menkaure", 12.4, 1.0, 14.1, 1.8),
    ("Sahure", 23, 10, None, None),
    ("Neferirkare", 30, 10, None, None),
]

print(
    f"{'Pyramid':<10} "
    f"{'Mean align.':>12} "
    f"{'σ_mean':>10}"
)
print("-" * 36)

for name, E, E_unc, W, W_unc in pyramids:
    if E is not None and W is not None:
        # Difference
        delta = E - W

        # Compute systematic uncertainty
        numerator = delta**2 - (E_unc**2 + W_unc**2)
        if numerator > 0:
            sigma_sys = math.sqrt(numerator / 2)
        else:
            sigma_sys = 0.0

        # Inflate uncertainties
        sigma_E_new = math.sqrt(E_unc**2 + sigma_sys**2)
        sigma_W_new = math.sqrt(W_unc**2 + sigma_sys**2)

        # Weighted mean
        weight_E = 1 / sigma_E_new**2
        weight_W = 1 / sigma_W_new**2

        mean_alignment = (E * weight_E + W * weight_W) / (weight_E + weight_W)
        sigma_mean = math.sqrt(1 / (weight_E + weight_W))

        print(
            f"{name:<10} "
            f"{mean_alignment:12.2f} "
            f"{sigma_mean:10.2f}"
        )
    else:
        print(
            f"{name:<10} "
            f"{'No data':>12} "
            f"{'-':>10}"
        )


# In[314]:


import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit

# ------------------------
# Data: Years after 2600 BC, alignment (arcmin), uncertainty
data = [
    (2600 - 2600, -19.0, 1.0),   # Meidum
    (2600 - 2583, -14.0, 3.0),   # Bent
    (2600 - 2554,  -3.1, 0.3),   # Khufu
    (2600 - 2522,   6.0, 0.1),   # Khafre
    (2600 - 2489,  12.8, 0.9),   # Menkaure
]

t = np.array([d[0] for d in data])      # forward time in years
align = np.array([d[1] for d in data])  # combined alignment
errors = np.array([d[2] for d in data]) # uncertainty on alignment

# ------------------------
# Linear model
def model(t, a, b):
    return a + b * t

# Chi-squared function
def chi2(a, b):
    return np.sum(((align - model(t, a, b)) / errors) ** 2)

# ------------------------
# Minuit fit
m = Minuit(chi2, a=0.0, b=0.274)
m.errordef = Minuit.LEAST_SQUARES
m.migrad()
m.hesse()

# Extract results
a_fit, b_fit = m.values["a"], m.values["b"]
a_err, b_err = m.errors["a"], m.errors["b"]

# Slope significance vs astronomical prediction
b_astro = 0.274
Z = (b_fit - b_astro) / b_err

# ------------------------
# Plot
t_plot = np.linspace(t.min(), t.max(), 200)
plt.errorbar(t, align, yerr=errors, fmt='o', capsize=4, color='black', label="Pyramids")
plt.plot(t_plot, model(t_plot, a_fit, b_fit), 'r--', label="Linear fit")

plt.xlabel("Years after 2600 BC")
plt.ylabel("Alignment (arc min)")
plt.title("Pyramid alignment vs years after 2600 BC")
plt.grid()
plt.legend()

# ------------------------
# Add text box with fit parameters
textstr = (
    f"Fit results:\n"
    f"a = {a_fit:.2f} ± {a_err:.2f} arcmin\n"
    f"b = {b_fit:.3f} ± {b_err:.3f} arcmin/year\n"
    f"Slope vs prediction: {b_astro:.3f} arcmin/year\n"
    f"Z = {Z:.2f} σ"
)

plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
plt.tight_layout()
plt.savefig('Problem 5.2.3 and 5.2.4.png', dpi = 300)
plt.show()


# In[311]:


import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from scipy.stats import chi2 as chi2_dist

# ------------------------
# Data: Years after 2600 BC, alignment (arcmin), uncertainty
data = [
    (2600 - 2600, -19.0, 1.0),   # Meidum
    (2600 - 2583, -14.0, 3.0),   # Bent
    (2600 - 2554,  -3.1, 0.3),   # Khufu
    (2600 - 2522,   6.0, 0.1),   # Khafre
    (2600 - 2489,  12.8, 0.9),   # Menkaure
]

t = np.array([d[0] for d in data])      # forward time in years
align = np.array([d[1] for d in data])  # combined alignment
errors = np.array([d[2] for d in data]) # sigma_mean

# ------------------------
# Linear model
def model(t, a, b):
    return a + b * t

# Chi-squared function
def chi2_func(a, b):
    return np.sum(((align - model(t, a, b)) / errors) ** 2)

# ------------------------
# Minuit fit
m = Minuit(chi2_func, a=0.0, b=0.0)
m.errordef = Minuit.LEAST_SQUARES
m.migrad()
m.hesse()

a_fit, b_fit = m.values["a"], m.values["b"]
a_err, b_err = m.errors["a"], m.errors["b"]

print("----- Fit Results -----")
print(f"Intercept a = {a_fit:.3f} ± {a_err:.3f} arcmin")
print(f"Slope b     = {b_fit:.4f} ± {b_err:.4f} arcmin/year")

# ------------------------
# Goodness-of-fit: chi-squared and p-value
residuals = align - model(t, a_fit, b_fit)
chi2_value = np.sum((residuals / errors) ** 2)
dof = len(align) - 2  # N - number of fit parameters
p_value = 1 - chi2_dist.cdf(chi2_value, df=dof)

print("\n----- Goodness-of-Fit -----")
print(f"Chi-squared: {chi2_value:.2f}")
print(f"Degrees of freedom: {dof}")
print(f"Chi-squared per dof: {chi2_value/dof:.2f}")
print(f"p-value: {p_value:.3f}")

# ------------------------
# Compare slope to astronomical prediction
b_astro = 0.274  # arcmin/year
Z = (b_fit - b_astro) / b_err
print("\n----- Comparison to Astronomical Prediction -----")
print(f"Astronomical slope: {b_astro:.3f} arcmin/year")
print(f"Difference: {b_fit - b_astro:.3f} arcmin/year")
print(f"Consistency test: Z = {Z:.2f} σ")

# ------------------------
# Plot
t_plot = np.linspace(t.min(), t.max(), 200)
plt.errorbar(t, align, yerr=errors, fmt='o', capsize=4, label="Pyramids")
plt.plot(t_plot, model(t_plot, a_fit, b_fit), 'r-', label="Linear fit")
plt.xlabel("Years after 2600 BC")
plt.ylabel("Alignment (arc min)")
plt.title("Pyramid alignment vs forward time")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:




