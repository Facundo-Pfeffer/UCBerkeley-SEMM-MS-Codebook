#!/usr/bin/env python3
"""
Combined viscoelastic dashboards for CEE231:
1. Step Response (from HW8)
2. Sinusoidal Response (original HW9)

This file imports and executes both dashboard generators.
"""

from visco_step import build_step_response_dashboard
from visco_sinusoidal import build_sinusoidal_dashboard


if __name__ == "__main__":
    print("\n" + "="*70)
    print("CEE231 - Combined Viscoelastic Dashboards")
    print("="*70 + "\n")
    
    # Generate Step Response Dashboard
    print("[1/2] Generating Step Response Dashboard...")
    fig_step = build_step_response_dashboard()
    fig_step.write_html("visco_dashboard_step.html", include_plotlyjs="cdn", auto_open=False)
    print("      [SUCCESS] Generated: visco_dashboard_step.html\n")
    
    # Generate Sinusoidal Response Dashboard
    print("[2/2] Generating Sinusoidal Response Dashboard...")
    fig_sin = build_sinusoidal_dashboard()
    fig_sin.write_html("sls_sinusoidal_dashboard.html", include_plotlyjs="cdn", auto_open=False)
    print("      [SUCCESS] Generated: sls_sinusoidal_dashboard.html\n")
    
    print("="*70)
    print("All dashboards generated successfully!")
    print("="*70)

