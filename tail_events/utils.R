library(POT)
library(ismev)
library(fGarch)
library(MLmetrics)
source("definition.R")
#library(fExtremes)

# Constants
DAY = 24
MONTH = DAY*30


fit_predParallel_garch11 = function(date, data.series, next.return, return.mean, return.sd, lowers, n=NULL, q_fit=NULL, qs=NULL) {
  prediction_i = c(date, next.return[1], return.mean[1], return.sd[1])
  fitted.model = garchFit(
    formula = ~ garch(1, 1),
    data = data.series,
    cond.dist = "norm",
    trace = FALSE
  )
  # Get standardized residuals
  model.residuals  = fGarch::residuals(fitted.model , standardize = TRUE)
  model.coef = coef(fitted.model)
  # Predict next value
  model.forecast = fGarch::predict(object = fitted.model, n.ahead = 1)
  model.mean = model.forecast$meanForecast # conditional mean
  model.sd = model.forecast$standardDeviation # conditional volatility
  
  for (j in 1:length(qs)) {
    q = qs[j]
    prob = 1-q
    lower = lowers[j]
    
    # Calculate Value-At-Risk
    model.var = var.normal(probs=prob,
                           mean=model.mean,
                           sd=model.sd)
    # Calculate Expected Shortfall
    model.es = es.normal(probs=prob, mean=model.mean, sd=model.sd)
    
    # Calculate proba
    Z_hat = ( lower - model.mean) / model.sd
    model.proba = pnorm(Z_hat)
    model.proba = 1 - model.proba
    
    q_data = c(lower, model.var, model.es, model.proba, model.mean , model.sd)
    
    prediction_i = c(prediction_i, q_data)
  }
  return (prediction_i)
}

fit_predParallel = function(date, data.series, next.return, return.sd, lowers, n=NULL, q_fit=NULL, qs=NULL) {
  prediction_i = c(date, next.return[1], return.sd[1])
  fitted.model = garchFit(
    formula = ~ arma(3, 1) + garch(1, 2),
    data = data.series,
    cond.dist = "QMLE",
    trace = FALSE
  )
  # Get standardized residuals
  model.residuals  = fGarch::residuals(fitted.model , standardize = TRUE)
  model.coef = coef(fitted.model)
  # Predict next value
  model.forecast = fGarch::predict(object = fitted.model, n.ahead = 1)
  model.mean = model.forecast$meanForecast # conditional mean
  model.sd = model.forecast$standardDeviation # conditional volatility
  
  # Fit gpd to residuals over threshold
  # Determine threshold
  EVTmodel.threshold = quantile(model.residuals, (1 - q_fit))
  
  # Fit GPD to residuals
  EVTmodel.fit = gpd.fit(
    xdat = model.residuals,
    threshold = EVTmodel.threshold,
    show = FALSE
  )
  # Extract scale and shape parameter estimates
  EVTmodel.scale = EVTmodel.fit$mle[1]
  EVTmodel.shape = EVTmodel.fit$mle[2]
  # Estimate quantiles
  Nu = EVTmodel.fit$nexc
  
  for (j in 1:length(qs)) {
    q = qs[j]
    prob = 1-q
    lower = lowers[j]
    
    # Calculate Value-At-Risk
    EVTmodel.zq = var.gpd(prob, EVTmodel.threshold, EVTmodel.scale, EVTmodel.shape, n, Nu)
    # EVTmodel.zq = POT::qgpd(prob, loc=EVTmodel.threshold, scale=EVTmodel.scale, shape=EVTmodel.shape, lambda = 1 - Nu/n)
    EVTmodel.var = model.mean + model.sd * EVTmodel.zq
    model.var = var.normal(probs=prob,
                           mean=model.mean,
                           sd=model.sd)
    # Calculate Expected Shortfall
    model.es = es.normal(probs=prob, mean=model.mean, sd=model.sd)
    EVTmodel.es = model.mean + model.sd * es.gpd(var=EVTmodel.zq,
                                                 threshold=EVTmodel.threshold,
                                                 scale=EVTmodel.scale,
                                                 shape=EVTmodel.shape)
    
    # Calculate proba
    Z_hat = ( lower - model.mean) / model.sd
    model.proba = pnorm(Z_hat)
    model.proba = 1 - model.proba
    
    if (Z_hat < EVTmodel.threshold){
      #print(paste('normal', model.mean, Z_hat, EVTmodel.threshold))
      EVTmodel.proba = model.proba
      #print(paste('normal', EVTmodel.proba))
    }else{
      #print(paste(model.mean, Z_hat, EVTmodel.threshold))
      EVTmodel.proba = POT::pgpd(Z_hat, loc=EVTmodel.threshold,
                                 scale= EVTmodel.scale, shape = EVTmodel.shape, lambda = (1 - (Nu/n)))
      EVTmodel.proba = 1 - EVTmodel.proba
      #print(EVTmodel.proba)
      #print(paste('final proba from pareto: ', EVTmodel.proba))
    }
    q_data = c(lower, EVTmodel.threshold, EVTmodel.var, EVTmodel.es,  EVTmodel.proba,
               model.var, model.es, model.proba, model.mean , model.sd, EVTmodel.zq)
    
    prediction_i = c(prediction_i, q_data)
  }
  return (prediction_i)
}