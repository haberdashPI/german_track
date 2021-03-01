data {
    int<lower=1> n; // number of observations
    int<lower=1> k; // number of predictors
    int<lower=0, upper=1> y[n]; // outcomes
    matrix[n,k] A; // predictiors
    real theta_prior;
    real r;
}

parameters {
    vector[k] theta;
}

model {
    theta ~ normal(0, theta_prior);
    y ~ bernoulli(inv_logit(A * theta)*(1-2*r)+r);
}

