#!/usr/bin/env python3
"""Test script to verify Problem 3 summary generation."""

import numpy as np
from pathlib import Path
from problem3_summary import (
    generate_html_template,
    generate_system_properties_section,
    generate_equations_section,
    generate_response_statistics_section
)

# Mock data for testing
mass_matrix = np.array([[1180, 0, 0], [0, 1180, 0], [0, 0, 910]])
mode_shapes = np.array([[0.00771, -0.01916, 0.02051],
                        [0.01755, -0.01331, -0.01903],
                        [0.02495, 0.01982, 0.00914]])
f_n = np.array([2.00, 7.20, 13.75])
omega_n = 2 * np.pi * f_n
zeta = np.array([0.0113, 0.0157, 0.0093])

# Mock analyzer object
class MockAnalyzer:
    def __init__(self):
        self.Gamma = np.array([0.5, 0.3, 0.2])
        self.M_eff = np.array([500, 300, 200])
        self.h_star = np.array([5.0, 6.0, 7.0])

analyzer = MockAnalyzer()

# Test HTML template
print("Testing HTML template...")
html = generate_html_template()
print(f"✓ HTML template generated ({len(html)} chars)")

# Test system properties
print("\nTesting system properties section...")
try:
    sys_props = generate_system_properties_section(
        mass_matrix, mode_shapes, f_n, omega_n, zeta, analyzer
    )
    print(f"✓ System properties section generated ({len(sys_props)} chars)")
    # Check for common issues
    if '\\\\' in sys_props:
        print("  ⚠ Warning: Found LaTeX line breaks in HTML (should be HTML tags)")
    if '<tr>' in sys_props and '<td>' in sys_props:
        print("  ✓ HTML table tags found")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test equations section
print("\nTesting equations section...")
try:
    equations = generate_equations_section(analyzer, np.array([3.5, 7.0, 10.5]))
    print(f"✓ Equations section generated ({len(equations)} chars)")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test statistics section
print("\nTesting statistics section...")
try:
    u = np.random.randn(3, 100) * 0.1
    u_ddot = np.random.randn(3, 100) * 10
    V_base = np.random.randn(100) * 5
    M_base = np.random.randn(100) * 50
    time = np.linspace(0, 10, 100)
    stats = generate_response_statistics_section(u, u_ddot, V_base, M_base, time)
    print(f"✓ Statistics section generated ({len(stats)} chars)")
    if '\\\\' in stats:
        print("  ⚠ Warning: Found LaTeX line breaks in HTML (should be HTML tags)")
    if '<tr>' in stats and '<td>' in stats:
        print("  ✓ HTML table tags found")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("Test complete!")


