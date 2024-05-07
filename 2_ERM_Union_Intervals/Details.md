# Programming Assignment 2: Union Of Intervals

## Overview

This assignment explores the hypothesis class of a finite union of disjoint intervals and the properties of the Empirical Risk Minimization (ERM) algorithm for this class within a binary classification context. The sample space \(X\) is [0,1], and we consider binary classifications \(Y = \{0,1\}\).

## Task Description

The hypothesis class comprises hypotheses defined by up to \(k\) disjoint intervals on the unit interval [0,1]. For each set of \(k\) disjoint intervals, we define a hypothesis function \(h_I(x)\) that predicts 1 if \(x\) falls within any of the intervals, and 0 otherwise. We evaluate these hypotheses using a true distribution and empirically from data.

### Key Components

- **intervals.py**: Implements the ERM algorithm for selecting the best \(k\) intervals minimizing empirical errors.
- **skeleton.py**: Contains the structure for the necessary functions and experiments to be implemented as described below.

### Tasks

#### (a) Minimum Error Hypothesis in \(H_{10}\)
Calculate the hypothesis in \(H_{10}\) with the smallest error based on the true distribution \(P\), where \(P[x,y] = P[y|x] \cdot P[x]\), and \(P[x]\) is uniform on [0,1].

#### (b) Error Calculation and Experiments
For \(k = 3\) and varying \(n = 10, 15, 20, ..., 100\), draw samples, run the ERM algorithm, calculate both empirical and true errors, and plot these errors as a function of \(n\).

#### (c) Best ERM Hypothesis for Varying \(k\)
Draw a large sample (\(n = 1500\)) and determine the best ERM hypothesis for \(k = 1, 2, ..., 10\). Plot the empirical and true errors as functions of \(k\) and identify the optimal \(k\) (denoted \(k^*\)).

#### (d) Structural Risk Minimization (SRM)
Using the SRM principle, determine a \(k\) that balances the empirical error and a complexity penalty for hypotheses up to \(k = 10\). Run experiments similar to those in (c) but include the penalty in the evaluations.

#### (e) Holdout-Validation
Using holdout-validation with 20% of the data, find a hypothesis for \(k = 1, ..., 10\) that potentially minimizes the true error. Discuss the efficacy of this method compared to previous approaches.

## Notes
Make sure your Python environment is set up with Python 3 and all necessary packages installed.
