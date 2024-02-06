# cartpole-knode-mpc
KNODE-MPC: Neural ODE based Predictive Controller for the Cartpole System.
Ved Radhakrishnan,
Conestoga High School, Berwyn, PA
Mentored by Mr. KongYao Chee, PhD student and Dr. George Pappas, Professor and Chair
ESE dept, University of Pennsylvania

This project introduces a hybrid Knowledge-based Neural ODE and Model Predictive Control
(KNODE-MPC) framework tailored for the dynamic control of a cartpole system, a classic
problem in robotics that exemplifies the challenges of stabilizing an inverted pendulum on a
moving cart. Recognizing the limitations of conventional model predictive control (MPC) in
handling system uncertainties and nonlinear dynamics, we implemented a hybrid modeling
approach that integrates a neural ordinary differential equations (NODE) model with traditional
physics-based models. This integration aims to capture the uncertainties inherent in the cartpole
system more accurately and data-efficiently than either approach alone.

By adopting a deep learning tool, specifically the KNODE, we augment the nominal physics-
based model of the cartpole system with a neural network learned from simulated data. This
hybrid model is then incorporated into a Linear Quadratic Regulator to form the KNODE-MPC,
designed to optimize the control actions over a prediction horizon while accounting for the
learned dynamics and uncertainties.

Preliminary validation of this model was conducted through extensive simulations with additional
introduced uncertainties. The results of the hybrid model demonstrate an enhancement in
trajectory tracking performance, highlighting its superior ability to maintain the stability of the
cartpole system under various operational conditions, including deviations of several
parameters from the nominal model.
The findings of this research will broadly impact the adaptive control of nonlinear systems like
the cartpole, and open new avenues for applying knowledge-based data-driven models to a
wide range of robotic control problems, such as control in quadrotor drone systems.
