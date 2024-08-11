# Uncertainty Estimation for Timeseries Forecasting
The Generalized Gauss-Newton method is used to estimate uncertainty in timeseries forecasting

In a deep neural network $f(x)$ the Hessian $\Psi$ of the loss with respect to the weights can give us a notion of uncertainty. Dimensions of the Hessian matrix, in which the model barely changes it's prediction when altering the weights, can thereby be considered as dimensions, in which the model is uncertain. However, the calculation of the Hessian is in $\mathcal{O}(ND^2)$ which means that for deep neural networks with a large number of parameters $D$ and a considerably high number of samples $N$ in the dataset the computation and storage of the Hessian matrix become infeasible.

There are several ways to approximate the Hessian matrix and lower the complexity of the computation. In this script the Generalized Gauss-Newton method is used to calculate the Hessian approximation and the uncertainty of the timeseries forecasting. 

<p align="center">
<img src="https://github.com/user-attachments/assets/b37b2548-19cf-4fb9-811f-3bde41ec7aa1" width="450"/>
</p>
