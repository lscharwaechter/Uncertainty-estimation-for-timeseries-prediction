# Uncertainty Estimation for Timeseries Prediction

In a deep neural network $f(x;\theta)$ the Hessian $\Psi$ of the loss with respect to the weights $\theta$ can give us a notion of uncertainty. Dimensions of the Hessian matrix, in which the model barely changes it's prediction when altering their values, can thereby be considered as dimensions, in which the model is uncertain. However, the calculation of the Hessian is in $\mathcal{O}(ND^2)$ which means that for deep neural networks with a large number of parameters $D$ and a considerably high number of samples $N$ in the dataset the computation and storage of the Hessian matrix become infeasible.

There are several ways to approximate the Hessian matrix and lower the complexity of the computation. In this script the **Generalized Gauss-Newton method** is used to calculate the Hessian approximation and the uncertainty of the timeseries prediction. The approximation is given by<p align="center">
  $\Psi \approx \alpha \mathcal{I} + GG^T$ </p>
  
where $GG^T$ is given by the matrix multiplication of the Jacobian of the loss with respect to the weights, $\alpha$ is a small constant and $\mathcal{I}$ is the identity matrix. The largest eigenvalues of the Hessian approximation then refer to the vector dimensions, in which a lot of change to the output happens when altering these values, i.e. contain certainty of the model. Ultimately, the inverse of the eigenvalues results in a notion of uncertainty. The figure below shows a possible representation of uncertainty in timeseries prediction using an LSTM model and a periodic signal, where the bounds are given by the standard deviation of the uncertainty. 

<p align="center">
<img src="https://github.com/user-attachments/assets/4616b0cb-e915-4db3-82aa-a215cc9c05dc" width="550"/>
</p>

This example is for illustrative purposes only. Alternatives for Uncertainty Estimation include Monte Carlo Dropout, Bootstrap Resampling, Monte Carlo sampling or diagonal approximation of the Fisher Information matrix, or Laplace approximation.
