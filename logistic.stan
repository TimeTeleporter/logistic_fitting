// Logistic curve with normally distributed measurement error:
//   y ~ normal(niveau / (1 + exp(-rate * (x - inflection))), sigma)

data {
  int<lower=1> N;            // number of observations
  vector[N] x;
  vector[N] y;

  int<lower=0> N_pred;       // number of points to predict the curve at
  vector[N_pred] x_pred;
}

parameters {
  real<lower=0> niveau;      // upper asymptote
  real<lower=0> rate;        // steepness
  real inflection;           // x value of the inflection point
  real<lower=0> sigma;       // measurement error
}

model {
  // Weakly informative priors
  niveau ~ normal(0, 5);
  rate ~ normal(0, 5);
  inflection ~ normal(0, 20);
  sigma ~ normal(0, 1);

  y ~ normal(niveau * inv_logit(rate * (x - inflection)), sigma);
}

generated quantities {
  // Fitted curve without measurement error
  vector[N_pred] mu_pred = niveau * inv_logit(rate * (x_pred - inflection));
}
