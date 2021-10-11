# Icpy

## Objective
calclate critical current (Ic) from experimental data for I - V curve autimatically.

## Requirements
- python 3.x
- numpy, scipy, matplotlib, seaborn (just for plot)

## How to caluclate Ic
1. Spline for experimental data using Scipy.
2. Delete baseline shifting caused by connection resistance, etc.
3. Use solver to y - Vc (Vc: electrical criterion), and define the last element of roots array as Ic.

