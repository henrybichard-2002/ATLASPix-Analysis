# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 14:20:58 2026

@author: henry
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_iec_plot():
    # Set publication quality plot parameters
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'lines.linewidth': 2.5
    })

    # Define the core physical parameters
    r_a = 0.20  # Anode radius in meters
    r_c = 0.05  # Cathode radius in meters
    v_0 = -50.0  # Cathode voltage in kilovolts
    v_virtual = 15.0  # Positive potential rise due to the central space charge

    # Calculate analytical constants for the vacuum potential region
    # The potential follows V(r) = A/r + B in a spherical vacuum
    constant_a = v_0 / (1/r_c - 1/r_a)
    constant_b = -constant_a / r_a

    # Generate the radial arrays for the distinct regions
    r_core = np.linspace(0, r_c, 500)
    r_vac = np.linspace(r_c, r_a, 500)
    r_out = np.linspace(r_a, 0.25, 100)

    # Calculate the electric potential for each region
    v_core = v_0 + v_virtual * (1 - (r_core / r_c)**2)
    v_vac = constant_a / r_vac + constant_b
    v_out = np.zeros_like(r_out)

    # Combine the arrays for a continuous plot
    r = np.concatenate((r_core, r_vac[1:], r_out[1:]))
    v = np.concatenate((v_core, v_vac[1:], v_out[1:]))

    # Initialize the figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the primary potential curve
    ax.plot(r * 100, v, color='navy', label='Electric Potential $V(r)$')

    # Highlight the physical grid structures
    ax.axvline(x=r_c * 100, color='red', linestyle='--', label='Cathode Grid')
    ax.axvline(x=r_a * 100, color='black', linestyle='--', label='Anode Chamber')

    # Add shading and annotations for the physics mechanisms
    ax.fill_between(r_core * 100, v_0, v_core, color='orange', alpha=0.3, label='Virtual Anode')
    ax.text(0, v_0 + v_virtual + 2, 'Central Charge', horizontalalignment='center', color='darkorange', fontsize=12)
    ax.text(r_c * 100 + 2, -25, 'Ion Acceleration Region', color='gray', fontsize=12)

    # Apply the final formatting touches
    ax.set_xlabel('Radius $r$ [cm]')
    ax.set_ylabel('Electric Potential $V$ [kV]')
    ax.set_title('IEC Reactor Electric Potential Profile')
    ax.grid(True, which='both', linestyle=':', alpha=0.7)
    ax.legend(loc='lower right')

    # Set the axis limits to frame the data neatly
    ax.set_ylim(v_0 - 5, 10)
    ax.set_xlim(0, 25)

    # Save the output file
    plt.savefig('iec_potential_profile.png', dpi=300, bbox_inches='tight')

generate_iec_plot()


import numpy as np
import matplotlib.pyplot as plt

def generate_cross_section_plot():
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'lines.linewidth': 2.5
    })

    energy_kev = np.logspace(0, 3, 500)

    def calculate_cross_section(energy, a1, a2, a3, a4, a5):
        numerator = a5 + a2 / ((a4 - a3 * energy)**2 + 1)
        denominator = energy * (np.exp(a1 / np.sqrt(energy)) - 1)
        return numerator / denominator

    sigma_dt = calculate_cross_section(energy_kev, 45.95, 50200, 1.368e-2, 1.076, 409)
    sigma_dd_pt = calculate_cross_section(energy_kev, 46.097, 372, 4.36e-3, 1.220, 0)
    sigma_dd_nhe = calculate_cross_section(energy_kev, 47.88, 482, 3.08e-3, 1.177, 0)

    sigma_dd_total = sigma_dd_pt + sigma_dd_nhe

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.loglog(energy_kev, sigma_dt, color='darkred', label='D-T Fusion')
    ax.loglog(energy_kev, sigma_dd_total, color='navy', label='D-D Fusion (Total)')
    ax.loglog(energy_kev, sigma_dd_nhe, color='royalblue', linestyle='--', label='D(d,n)$^3$He Branch')
    ax.loglog(energy_kev, sigma_dd_pt, color='deepskyblue', linestyle=':', label='D(d,p)T Branch')

    ax.set_xlabel('Incident Ion Energy [keV]')
    ax.set_ylabel('Cross Section $\sigma$ [barns]')
    ax.set_title('D-T and D-D Fusion Cross Sections')

    ax.grid(True, which='both', linestyle=':', alpha=0.7)
    ax.legend(loc='lower right')

    ax.set_xlim(1, 1000)
    ax.set_ylim(1e-6, 10)

    plt.savefig('fusion_cross_sections_corrected.png', dpi=300, bbox_inches='tight')

generate_cross_section_plot()

import numpy as np
import matplotlib.pyplot as plt

# Define energy array with exactly one bin per integer value
E = np.arange(1, 201, 1)

# Quantum Tunneling probability
b_DT = 34.38
b_DD = 31.39

P_DT = np.exp(-b_DT / np.sqrt(E))
P_DD = np.exp(-b_DD / np.sqrt(E))

# Nuclear Resonances 
S_DT = 1.0e4 * (40.0**2 / ((E - 64.0)**2 + 40.0**2)) + 1.2e4
S_DD_p = 56.0 + 0.5 * E
S_DD_n = 54.0 + 0.5 * E

# Resultant Cross-Sections
sigma_DT = (S_DT / E) * P_DT
sigma_DD_p = (S_DD_p / E) * P_DD
sigma_DD_n = (S_DD_n / E) * P_DD

# Maxwell-Boltzmann Distribution and Gamow Peak
T_ion = 20.0
MB_dist = (2.0 * np.sqrt(E / np.pi) / (T_ion**1.5)) * np.exp(-E / T_ion)

gamow_DT = MB_dist * np.sqrt(E) * sigma_DT
gamow_DD_p = MB_dist * np.sqrt(E) * sigma_DD_p
gamow_DD_n = MB_dist * np.sqrt(E) * sigma_DD_n

# Integrate the area under the Gamow peaks
integral_DT = np.trapz(gamow_DT, E)
integral_DD_p = np.trapz(gamow_DD_p, E)
integral_DD_n = np.trapz(gamow_DD_n, E)

print(f"Total theoretical rate for D-T: {integral_DT:.2f}")
print(f"Total theoretical rate for D-D(p): {integral_DD_p:.2f}")
print(f"Total theoretical rate for D-D(n): {integral_DD_n:.2f}")

# Create publication quality plots
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'lines.linewidth': 2.5
})

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Subplot 1
axs[0, 0].plot(E, P_DT, label='D-T Tunneling', color='blue')
axs[0, 0].plot(E, P_DD, label='D-D Tunneling', color='red', linestyle='--')
axs[0, 0].set_title('Quantum Tunneling Probability')
axs[0, 0].set_ylabel('Probability')
axs[0, 0].set_yscale('log')
axs[0, 0].set_xscale('log')
axs[0, 0].legend()
axs[0, 0].grid(True, alpha=0.3)

# Subplot 2
axs[0, 1].plot(E, S_DT, label='D-T S-factor', color='blue')
axs[0, 1].plot(E, S_DD_p, label='D-D(p) S-factor', color='green', linestyle='-.')
axs[0, 1].plot(E, S_DD_n, label='D-D(n) S-factor', color='orange', linestyle=':')
axs[0, 1].set_title('Nuclear Resonances (S-factor)')
axs[0, 1].set_ylabel('S-factor (keV*barn)')
axs[0, 1].set_yscale('log')
axs[0, 1].set_xscale('log')
axs[0, 1].legend()
axs[0, 1].grid(True, alpha=0.3)

# Subplot 3
axs[1, 0].plot(E, sigma_DT, label='D-T Cross-Section', color='blue')
axs[1, 0].plot(E, sigma_DD_p, label='D-D(p) Cross-Section', color='green', linestyle='-.')
axs[1, 0].plot(E, sigma_DD_n, label='D-D(n) Cross-Section', color='orange', linestyle=':')
axs[1, 0].set_title('Resultant Cross-Sections')
axs[1, 0].set_xlabel('Energy (keV)')
axs[1, 0].set_ylabel('Cross-Section (barns)')
axs[1, 0].set_yscale('log')
axs[1, 0].set_xscale('log')
axs[1, 0].legend()
axs[1, 0].grid(True, alpha=0.3)

# Subplot 4
axs[1, 1].plot(E, gamow_DT, label='D-T Gamow', color='blue')
axs[1, 1].plot(E, gamow_DD_p, label='D-D(p) Gamow', color='green', linestyle='-.')
axs[1, 1].plot(E, gamow_DD_n, label='D-D(n) Gamow', color='orange', linestyle=':')
axs[1, 1].set_title('Reaction Probability (Gamow Peaks)')
axs[1, 1].set_xlabel('Energy (keV)')
axs[1, 1].set_ylabel('Probability Rate')
axs[1, 1].set_yscale('log')
axs[1, 1].set_ylim(bottom=1e-10) 
axs[1, 1].legend()
axs[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gamow_integration_log.png', dpi=300)