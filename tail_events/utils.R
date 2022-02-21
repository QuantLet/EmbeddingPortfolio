library(fGarch)
library(xts)
library(forecast)
library(parallel)
library(doParallel)

# library(POT)
# library(ismev)
# library(MLmetrics)
# source("definition.R")
#library(fExtremes)

get_cv_data = function(path, start_date, test_start, end_date, window_size = NULL) {
  # load dataset
  data = load_data(path = path, start_date = start_date, end_date = end_date)
  train_data = data[index(data) < test_start]
  if (!is.null(window_size)) {
    start_date = index(train_data)[nrow(train_data) - config$window_size + 1]
    train_data = train_data[start_date <= index(train_data),]
  }
  test_data = data[test_start <= index(data)]

  return(list(train = train_data, test = test_data))
}

load_data = function(path, end_date = NULL, start_date = NULL, window_size = NULL) {
  data = read.zoo(path, index.column = 1, dec = ".", sep = ",",
                  format = "%Y-%m-%d", read = read.csv)
  data = as.xts(data)
  if (!is.null(end_date)) {
    data = data[index(data) <= end_date,]
  }
  if (!is.null(window_size)) {
    start_date = index(data)[length(index(data)) - window_size + 1]
  }
  if (!is.null(start_date)) {
    data = data[start_date <= index(data),]
  }
  return(data)
}

fit_model = function(data, cond.dist, p = NULL, q = NULL, formula = NULL) {
  n = nrow(data)
  # Select mean model
  if (is.null(formula)) {
    ARIMAfit = forecast::auto.arima(data, method = "CSS-ML", start.p = 1, start.q = 1, seasonal = FALSE) # stepwise=FALSE, parallel=TRUE, num.cores = parallel::detectCores() - 1)
    arima.order = unname(forecast::arimaorder(ARIMAfit))
    n_nans = 0
    if (arima.order[2] > 0) {
      data = diff(data, lag = 1, differences = arima.order[2], na.pad = FALSE)
      n_nans = n - nrow(data)
      ARIMAfit = forecast::auto.arima(data, method = "CSS-ML", start.p = 1, start.q = 1, seasonal = FALSE)  # stepwise=FALSE, parallel=TRUE, num.cores = parallel::detectCores() - 1)
      arima.order = unname(forecast::arimaorder(ARIMAfit))
    }
    stopifnot(arima.order[2] == 0)
    formula = substitute(~arma(a, b) + garch(p, q),
                         list(a = arima.order[1], b = arima.order[3], p = p, q = q))
  }
  garch.model = tryCatch(
  {
    garch.model = fGarch::garchFit(
      formula = formula,
      data = data,
      cond.dist = cond.dist,
      # algorithm = "lbfgsb",
      trace = FALSE)
    return(list(model = garch.model, aic = garch.model@fit$ics[1]))
  },
    error = function(e)
    {
      message(e)
      return(list(model = NULL, aic = Inf))
    },
    silent = FALSE)
}


model_selection = function(data, model.params, fit_model, parallel = TRUE) {
  # Normalize data to have variance 1
  data.sd = sd(data)
  data = data / data.sd
  tuning.grid = expand.grid(
    cond.dist = model.params$cond.dist,
    garch.order.p = model.params$garch.order.p,
    garch.order.q = model.params$garch.order.q,
    stringsAsFactors = FALSE
  )

  if (parallel) {
    # Get number of cores
    n.cores = parallel::detectCores() - 1
    #create the cluster
    my.cluster = parallel::makeCluster(
      n.cores
    )
    #register it to be used by %dopar%
    doParallel::registerDoParallel(cl = my.cluster)
    # doParallel::clusterCall(cl = my.cluster, function() source("utils.R"))
    #check if it is registered (optional)
    stopifnot(foreach::getDoParRegistered())
    # How many workers are availalbes ?
    # print(paste(foreach::getDoParWorkers()," workers available"))

    result = foreach(
      cond.dist = tuning.grid$cond.dist,
      garch.order.p = tuning.grid$garch.order.p,
      garch.order.q = tuning.grid$garch.order.q,
      .packages = c("forecast", "fGarch")
    ) %dopar% {

      tryCatch(
        fit_model(data, cond.dist, p = garch.order.p, q = garch.order.q),
        error = function(e) list(model = NULL, aic = Inf),
        silent = FALSE)
    }
    parallel::stopCluster(cl = my.cluster)
  } else {
    result = list()
    for (i in 1:nrow(tuning.grid)) {
      cond.dist = tuning.grid[i, "cond.dist"]
      garch.order.p = tuning.grid[i, "garch.order.p"]
      garch.order.q = tuning.grid[i, "garch.order.q"]
      r = fit_model(data, cond.dist, p = garch.order.p, q = garch.order.q)
      result = append(result, list(r))
    }
  }

  AIC = c()
  for (i in 1:length(result)) {
    AIC = c(AIC, result[[i]]$aic)
  }
  best_model = result[[which.min(AIC)]]$model
  aic = result[[which.min(AIC)]]$aic

  return(list(model = best_model, aic = aic))
}

predict_proba = function(train_data, test_data, window_size, model,
                         fit_model, next_proba, parallel = TRUE) {

  formula = model@formula
  cond.dist = model@fit$params$cond.dist

  if (parallel) {
    # Get number of cores
    n.cores = parallel::detectCores() - 1
    #create the cluster
    my.cluster = parallel::makeCluster(
      n.cores
    )
    #register it to be used by %dopar%
    doParallel::registerDoParallel(cl = my.cluster)
    # doParallel::clusterCall(cl = my.cluster, function() source("utils.R"))
    #check if it is registered (optional)
    stopifnot(foreach::getDoParRegistered())
    # How many workers are availalbes ?
    # print(paste(foreach::getDoParWorkers()," workers available"))
    probas = foreach(
      i = 1:nrow(test_data),
      .combine = 'c',
      .packages = c("forecast", "fGarch")
    ) %dopar% {
      if (i == 1) {
        proba = tryCatch(
          next_proba(model),
          error = function(e) NaN,
          silent = FALSE
        )

      } else {
        temp = rbind(train_data, test_data[1:(i - 1),])
        temp = tail(temp, window_size)
        # Normalize data to have unit variance
        temp = temp / sd(temp)
        model = tryCatch(
          fit_model(temp, cond.dist, formula = formula),
          error = function(e) list(model = NULL, aic = Inf),
          silent = FALSE)
        model = model$model
        if (!is.null(model)) {
          proba = tryCatch(
            next_proba(model),
            error = function(e) NaN,
            silent = FALSE
          )
        } else {
          proba = NaN
        }
      }
      proba
    }
    parallel::stopCluster(cl = my.cluster)
  } else {
    probas = c()
    for (i in 1:nrow(test_data)) {
      if (i == 1) {
        if (!is.null(model)) {
          proba = next_proba(model)
        } else {
          proba = NaN
        }
      } else {
        temp = rbind(train_data, test_data[1:(i - 1),])
        temp = tail(temp, window_size)
        # Normalize data to have unit variance
        temp = temp / sd(temp)
        model = fit_model(temp, cond.dist, formula = formula)
        model = model$model
        if (!is.null(model)) {
          proba = next_proba(model)
        } else {
          proba = NaN
        }
      }
      probas = c(probas, proba)
    }
  }
  # probas = xts(probas, order.by = index(test_data))
  # colnames(probas) = "proba"

  return(probas)
}

next_proba = function(object, conf = 0.95) {
  cond.dist = object@fit$params$cond.dist
  # Predict next value
  model.forecast = fGarch::predict(object = object, n.ahead = 1, conf = conf)
  meanForecast = model.forecast$meanForecast # conditional mean from mean model
  meanError = model.forecast$meanError # Error
  sdForecast = model.forecast$standardDeviation # conditional volatility
  # Get conf interval:
  # https://rdrr.io/cran/fGarch/src/R/methods-predict.R

  if (cond.dist == "norm") {
    crit_valu = qnorm(1 - (1 - conf) / 2)
    crit_vald = qnorm((1 - conf) / 2)
  }
  if (cond.dist == "snorm") {
    crit_valu = qsnorm(1 - (1 - conf) / 2, xi = coef(object)["skew"])
    crit_vald = qsnorm((1 - conf) / 2, xi = coef(object)["skew"])
  }
  if (cond.dist == "std") {
    crit_valu = fGarch::qstd(1 - (1 - conf) / 2, nu = coef(object)["shape"])
    crit_vald = fGarch::qstd((1 - conf) / 2, nu = coef(object)["shape"])
  }
  if (cond.dist == "sstd") {
    crit_valu = fGarch::qsstd(1 - (1 - conf) / 2, nu = coef(object)["shape"],
                              xi = coef(object)["skew"])
    crit_vald = fGarch::qsstd((1 - conf) / 2, nu = coef(object)["shape"],
                              xi = coef(object)["skew"])
  }
  if (cond.dist == "QMLE") {
    e = sort(object@residuals / object@sigma.t)
    crit_valu = e[round(t * (1 - (1 - conf) / 2))]
    crit_vald = e[round(t * (1 - conf) / 2)]
  }

  int_l = meanForecast + crit_vald * meanError
  int_u = meanForecast + crit_valu * meanError

  # Calculate proba
  Z_hat = -meanForecast / sdForecast
  if (cond.dist == "norm") {
    proba = pnorm(Z_hat)
  }
  if (cond.dist == "snorm") {
    proba = psnorm(Z_hat)
  }
  if (cond.dist == "std") {
    proba = fGarch::pstd(Z_hat)
  }
  if (cond.dist == "sstd") {
    proba = fGarch::psstd(Z_hat)
  }

  return(proba)
}

get_dist_functon = function(dist) {
  if (dist == "norm") {
    func = pnorm
  }
  if (dist == "snorm") {
    func = psnorm
  }
  if (dist == "std") {
    func = fGarch::pstd
  }
  if (dist == "sstd") {
    func = fGarch::psstd
  }

  return(func)
}