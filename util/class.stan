data{
  int<lower=1> n_t; // number of time samples
  int<lower=1> n_k; // number of classes
  int<lower=1> n_e; // the number of possible envelopes
  int<lower=1> d_y; // number of y dimensions
  int<lower=1> d_x; // number of x dimensions

  matrix[n_t,d_y] y; // the response to classify/predict
  matrix[n_t,d_x] x[n_e]; // the possible envelopes the response could be encoding
  matrix[n_t] k; // the label for each time point

  real sigma_y; // plausible error in y
  real sigma_W; // plausible error in weighting between x and y
}
parameters{
  matrix[n_y,n_x] W[n_e,n_k]; // the transformation from x -> y for each class
  real eps; // error in y
  real phi[n_e]; // error in W for each envelope
}
model{
  eps ~ normal(0,sigma_y);
  phi ~ normal(0,sigma_W);

  for(env in 1:n_e){
    for(k in 1:n_k){}
      W[env,k] ~ double_exponential(0,phi[env]);
  }
  for(i in 1:n_t){
    for(env in 1:n_e){
      y[i] ~ normal(W[env,k[i]]*x[env][i],eps);
    }
  }
}

// NOTES: I think x in this case is a set of possible envelopes
// i.e. the mixture, the target, the non-target, etc....
// this way, the model can learn which of the envelopes is
// most informative for each class k

// NEXT STEP: create a toy data set and test the model
