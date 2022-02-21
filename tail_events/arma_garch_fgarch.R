rm(list = ls(all = TRUE))
graphics.off()

# install and load packages
libraries = c("fGarch", "FinTS", "forecast", "xts")
lapply(libraries, function(x) if (!(x %in% installed.packages())) {
  install.packages(x)
})
lapply(libraries, library, quietly = TRUE, character.only = TRUE)

load_data = function(path, end_date=NULL, start_date=NULL, window_size=NULL) {
  data <- read.zoo(path, index.column=1, dec=".", sep=",",
                   format="%Y-%m-%d", read=read.csv)
  data <- as.xts(data)
  if (!is.null(end_date)) {
    data = data[index(data)<=end_date, ]
  }
  if (!is.null(window_size)) {
   start_date = index(data)[length(index(data)) - window_size + 1]
  }
  print(start_date)
  print(data[start_date<=index(data), ])
  if (!is.null(start_date)) {
    data = data[start_date<=index(data), ]
  }
  return (data)
}

# load dataset
data = load_data(path="./data/dataset1/train_linear_activation.csv",
                 end_date="2018-07-10",
                 window_size=250)
index = index(data)
factors = colnames(data)
ind = 2
factor.name = factors[ind]
fdata = data[,ind]
head(fdata)
# Plot data
plot(fdata, ylab=factor.name)


# Fit ARIMA model
X = fdata /sd(fdata)# 100 * log(sp500ret + 1) # fdata /sd(fdata)#- mean(fdata)
n = length(index(X))
fit <- auto.arima(X)
arima.order = arimaorder(fit)
print(arima.order)
ARIMAfit <- arima(X, order = arima.order)
summary(ARIMAfit)
par(mfrow = c(1, 1))
mu = xts(fitted(ARIMAfit), order.by=index(fdata))
res = xts(ARIMAfit$residuals, order.by=index(fdata))
plot(X, type="l",col="green")
lines(mu, col="red")

plot(X, type="l",col="green")
lines(res, col="red")

# Get residuals and check for volatility cluster
par(mfrow = c(1, 1))
plot(res, ylab = NA, type = 'l')
# Check the squared residuals for conditional heteroscedasticity: the ARCH effects.
res2 = res^2
plot(res2, ylab='Squared residuals', main=NA)

# Test the ARCH effect: p value close to 0 => ARCH effect
ArchTest(res, lag=12)  #library FinTS
Box.test(res2, type = "Ljung-Box", lag=12)
# We reject null hypothesis of both Archtest and Ljung-Box => autocorrelation in the squared residuals

# Plot the autocorrelation of the squared residuals
par(mfrow = c(1, 2))
acfres2 = acf(res2, main = NA, lag.max = 20, ylab = "Sample Autocorrelation", 
              lwd = 2)
pacfres2 = pacf(res2, lag.max = 20, ylab = "Sample Partial Autocorrelation", 
                lwd = 2, main = NA)
# We see some autocorrelation until lag 3 on the PACF we could specify a ARCH(3) model

# Standard GARCH
cond.dist = "norm"
garch.model = garchFit(
  formula = ~ arma(0, 1) + garch(1, 1),
  data = X,
  cond.dist = cond.dist,
  trace = FALSE
)
garch.model

# Get standardized residuals
model.fitted = xts(garch.model@fitted, order.by=index(fdata))
model.sigma = xts(garch.model@sigma.t, order.by=index(fdata))
model.residuals = xts(fGarch::residuals(garch.model, standardize = FALSE), order.by=index(fdata))
model.coef = coef(garch.model)


par(mfrow = c(1, 1))
plot(fdata - model.fitted, type="l")
lines(model.residuals)

Z_hat = 
plot(Z_hat)


par(mfrow = c(1, 1))
plot(X, type="l",col="green")
lines(model.fitted, col="red")
lines(model.residuals, col="blue")
lines(X-model.fitted, col="brown")

# Predict next value
conf = 0.95
model.forecast = fGarch::predict(object = garch.model, n.ahead = 1, conf=conf)
model.mean = model.forecast$meanForecast # conditional mean from mean model
model.meanError = model.forecast$meanError # Error
model.sd = model.forecast$standardDeviation # conditional volatility
# Get conf interval:
# https://rdrr.io/cran/fGarch/src/R/methods-predict.R
if (cond.dist == "norm") {
  crit_valu <- qnorm(1-(1-conf)/2)
  crit_vald <- qnorm((1-conf)/2)
}
int_l <- model.mean+crit_vald*meanError
int_u <- meanForecast+crit_valu*meanError

# Calculate proba
# For garch: Z_t = (x_t - mu_t)/sigma_t so x_t < 0 <=>  - mu_t/sigma_t < 0
Z_hat = - model.mean / model.sd
if (cond.dist == "norm") {
  model.proba = pnorm(Z_hat)
}
print(model.proba)

