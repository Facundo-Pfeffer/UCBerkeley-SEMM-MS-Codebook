#!/usr/bin/env python3
"""
Combined viscoelastic dashboards for CEE231:
1. Step Response (from HW8)
2. Sinusoidal Response (original HW9)

This file imports and executes both dashboard generators.
"""

from pathlib import Path

from visco_step import build_step_response_dashboard
from visco_sinusoidal import build_sinusoidal_dashboard

HW_DIR = Path(__file__).resolve().parent


if __name__ == "__main__":
    print("\n" + "="*70)
    print("CEE231 - Combined Viscoelastic Dashboards")
    print("="*70 + "\n")
    
    # Generate Step Response Dashboard
    print("[1/2] Generating Step Response Dashboard...")
    fig_step = build_step_response_dashboard()
    step_output = HW_DIR / "visco_dashboard_step.html"
    fig_step.write_html(step_output, include_plotlyjs="cdn", auto_open=False)
    print(f"      [SUCCESS] Generated: {step_output}\n")
    
    # Generate Sinusoidal Response Dashboard
    print("[2/2] Generating Sinusoidal Response Dashboard...")
    fig_sin = build_sinusoidal_dashboard()
    sin_output = HW_DIR / "sls_sinusoidal_dashboard.html"
    fig_sin.write_html(sin_output, include_plotlyjs="cdn", auto_open=False)
    print(f"      [SUCCESS] Generated: {sin_output}\n")
    
    print("="*70)
    print("All dashboards generated successfully!")
    print("="*70)

