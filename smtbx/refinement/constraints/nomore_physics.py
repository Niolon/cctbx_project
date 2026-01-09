"""
Physics functions for NoMoRe constraint (Python fallback).

These are used for pre-computing HIGH mode contributions in Python.
The main computation uses the C++ implementation in nomore.cpp.
"""
from __future__ import absolute_import, division, print_function

import math

# Physical constants (same as C++ nomore.h, derived from scipy.constants)
K_TO_CM1 = 0.6950348004861275
U_CART_FACTOR_CM1 = 33.71525833617554


def bose_einstein_energy_cm1(freq_cm1, temperature_K):
  """Calculate Bose-Einstein energy in cm^-1."""
  kt_cm1 = temperature_K * K_TO_CM1
  if kt_cm1 < 1e-6:
    kt_cm1 = 1e-6

  x = freq_cm1 / kt_cm1

  if x < 1e-4 and x > 0:
    occupation = 1.0/x - 0.5
  elif x <= 0:
    occupation = 0.0
  else:
    occupation = 1.0 / (math.exp(x) - 1.0)

  return freq_cm1 * (0.5 + occupation)


def calculate_adp_amplitude(freq_cm1, temperature_K, min_freq=5.0):
  """Calculate ADP amplitude factor from frequency."""
  safe_freq = max(freq_cm1, min_freq)
  energy = bose_einstein_energy_cm1(safe_freq, temperature_K)
  return energy / (safe_freq * safe_freq) * U_CART_FACTOR_CM1
