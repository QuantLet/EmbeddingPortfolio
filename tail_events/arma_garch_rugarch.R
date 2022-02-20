rm(list = ls(all = TRUE))
graphics.off()

# install and load packages
libraries = c("rugarch", "FinTS", "forecast")
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
X = fdata/sd(fdata)# 100 * log(sp500ret + 1) # fdata /sd(fdata)#- mean(fdata)
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

# Standard GARCH: tune parameters
# mean.model = list(armaOrder=c(arima.order["p"], arima.order["q"]),
#                   arfima=TRUE,  include.mean = TRUE)
armaOrder <- c(0,1) # ARMA order
garchOrder <- c(1,1) # GARCH order
distribution.model = "norm"
mean.model = list(armaOrder = c(0,1))
varModel <- list(model = "sGARCH", garchOrder = garchOrder)
spec <- ugarchspec(varModel, mean.model = list(armaOrder = armaOrder), distribution.model = distribution.model) 
garch.model <- ugarchfit(spec, X, solver="hybrid")
print(infocriteria(garch.model)[1])

conf = 0.95

model.forecast = ugarchforecast(garch.model, n.ahead=1)
model.mean = model.forecast@forecast$seriesFor
model.sd = model.forecast@forecast$sigmaFor
# model.meanError = model.forecast$meanError # Error

# Calculate proba
# For garch: Z_t = (x_t - mu_t)/sigma_t so x_t < 0 <=>  - mu_t/sigma_t < 0
Z_hat = - model.mean / model.sd
if (distribution.model == "norm") {
  model.proba = pnorm(Z_hat)
}
print(model.proba)



par(mfrow = c(1, 1))
plot(X, type="l",col="green")
lines(fitted(garch.model), col="red")
lines(residuals(garch.model), col="blue")
lines(fitted(garch.model) + residuals(garch.model), col="red")


predict(garch.model)

# qq plot
par(pty="s")
plot(garch.model, which = 9)#, xlim = c(-15,15))


# To control plot param need to call qdist and .qqLine
zseries = as.numeric(residuals(garch.model, standardize=TRUE))
distribution = garch.model@model$modeldesc$distribution
idx = garch.model@model$pidx
pars  = garch.model@fit$ipars[,1]
skew  = pars[idx["skew",1]]
shape = pars[idx["shape",1]]
if(distribution == "ghst") ghlambda = -shape/2 else ghlambda = pars[idx["ghlambda",1]]

par(mfrow = c(1, 1), pty="s") 
n = length(zseries)
x = qdist(distribution = distribution, lambda = ghlambda, 
          skew = skew, shape = shape, p = ppoints(n))[order(order(zseries))]
plot(x, zseries,  ylim = c(-4, 4), ylab="Sample Quantiles", xlab="Theoretical Quantiles")
rugarch:::.qqLine(y = zseries, dist = distribution, datax = TRUE,  lambda = ghlambda, 
                  skew = skew, shape = shape)

plot(fitted(garch.model))
plot(fdata)
