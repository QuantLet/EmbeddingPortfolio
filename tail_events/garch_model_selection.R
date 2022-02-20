rm(list = ls(all = TRUE))
graphics.off()

# https://cran.r-project.org/web/packages/qrmtools/vignettes/ARMA_GARCH_VaR.html

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
   start_date = index(data)[length(index(data)) - window_size]
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
plot(data)

ind = 2
factor.name = factors[2]
fdata = data[,ind]
head(fdata)
# Plot data
plot(fdata, ylab=f)

par(mfrow = c(1, 2))
# histogram of returns
hist(fdata, col = "grey", breaks = 40, freq = FALSE, xlab = NA)
lines(density(fdata), lwd = 2)
mu = mean(fdata)
sigma = sd(fdata)
x = seq(-4, 4, length = 100)
curve(dnorm(x, mean = mu, sd = sigma), add = TRUE, col = "darkblue", 
      lwd = 2)
# qq-plot
par(pty="s") 
d = (coredata(fdata) - mu)/ sigma
plot(qnorm(seq(0,1, length.out=length(d))), d[order(d)], xlim = c(-4,4), ylim = c(-4,4), main = NULL,
     ylab = "Sample Quantiles", xlab = "Theoretical Quantiles")

lines(qnorm(seq(0,1, length.out=length(d))), qnorm(seq(0,1, length.out=length(d))))
lines(seq(-15,
          qnorm(seq(0,1, length.out=length(d)))[2],
          length.out=length(d)
),
seq(-15,
    qnorm(seq(0,1, length.out=length(d)))[2],
    length.out=length(d)
))

lines(seq(qnorm(seq(0,1, length.out=length(d)))[length(d) - 1],
          15,
          length.out=length(d)
),
seq(qnorm(seq(0,1, length.out=length(d)))[length(d) - 1],
    15,
    length.out=length(d)
))
par(pty="s") 
plot(qnorm(seq(0,1, length.out=length(d))),
     d[order(d)], xlim = c(-4,4), ylim = c(-4,4), main = NULL,
     ylab = "Sample Quantiles", xlab = "Theoretical Quantiles")
abline(coef = c(0,1))
qqnorm(d, xlim = c(-4,4), ylim = c(-4,4), main = NULL)
qqline(d, probs = c(0.01, 0.99))


# Methodology for ARCH model
# 1. Specify a mean equation by testing for serial dependence in the data and, if
# necessary, building an econometric model (e.g., an ARMA model) for the
# return series to remove any linear dependence.
# 2. Use the residuals of the mean equation to test for ARCH effects.
# 3. Specify a volatility model if ARCH effects are statistically significant, and
# perform a joint estimation of the mean and volatility equations.
# 4. Check the fitted model carefully and refine it if necessary.


# Fit ARIMA model
X = fdata
fit <- auto.arima(X)
arima.order = arimaorder(fit)
print(arima.order)
ARIMAfit <- arima(X, order = arima.order)
summary(ARIMAfit)

# Get residuals and check for volatility cluster
par(mfrow = c(1, 1))
res = ARIMAfit$residuals
plot(res, ylab = NA, type = 'l')
# Check the squared residuals for conditional heteroscedasticity: the ARCH effects.
res2 = ARIMAfit$residuals^2
plot(res2, ylab='Squared residuals', main=NA)

# Test the ARCH effect: p value close to 0 => ARCH effect
res = ARIMAfit$residuals
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
mean.model = list(armaOrder=c(arima.order["p"], arima.order["q"]), arfima=TRUE)
# #fit the rugarch eGarch model with skew student distribution
# spec = ugarchspec(mean.model = mean.model,
#                   variance.model = list(model = 'sGARCH', garchOrder = c(1,0)),
#                   distribution = 'norm',
#                   fixed.pars=list(arfima=arima.order["d"]))
# 

dist_params = c("norm", "snorm", "std", "sstd")
result <- data.frame(matrix(ncol = 3, nrow = length(dist_params)))
colnames(result) = c("distribution", "order", "aic")
d=0
for (distribution in dist_params){
  print(distribution)
  d = d+1
  order_params = list(c(1,0), c(0,1), c(1,1), c(1,2), c(2,1), c(2,2))
  temp_res <- data.frame(matrix(ncol = 2, nrow = length(order_params)))
  colnames(temp_res) = c("order", "aic")
  c = 0
  for (garchOrder in order_params) {
    print(garchOrder)
    c = c +1
    temp_res$order[c] = paste(garchOrder, collapse = " ")
    spec = ugarchspec(mean.model = mean.model,
                      variance.model = list(model = 'sGARCH', garchOrder = garchOrder),
                      distribution = distribution,
                      fixed.pars=list(arfima=arima.order["d"]))
    
    garch.model <- ugarchfit(spec, X)
    temp_res$aic[c] = infocriteria(garch.model)[1]
  }
  best.aic = min(temp_res$aic)
  garchOrder = temp_res$order[which.min(temp_res$aic)]
  result$distribution[d] = distribution
  result$order[d] = garchOrder
  result$aic[d] = best.aic
}

print(result)
distribution = result$distribution[which.min(result$aic)]
print("Final distribution")
print(distribution)
garchOrder = result$order[which.min(result$aic)]
garchOrder = c(strtoi(substr(garchOrder,1,1)), strtoi(substr(garchOrder,3,3)))
print("Final GARCH order") 
print(garchOrder)

spec = ugarchspec(mean.model = mean.model,
                  variance.model = list(model = 'sGARCH', garchOrder = garchOrder),
                  distribution = distribution,
                  fixed.pars=list(arfima=arima.order["d"]))
garch.model <- ugarchfit(spec, X)
print(infocriteria(garch.model)[1])

# Now compare with ARCH model defined with PCAF
spec = ugarchspec(mean.model = mean.model,
                  variance.model = list(model = 'sGARCH', garchOrder = c(3,0)),
                  distribution = distribution,
                  fixed.pars=list(arfima=arima.order["d"]))
arch.model <- ugarchfit(spec, X)
print(infocriteria(arch.model)[1])
print(min(result$aic))
# We prefer the GARCH model


par(mfrow = c(1, 1))
plot(X, type="l",col="green")
lines(fitted(garch.model), col="red")
lines(residuals(garch.model) + mean(X), col="blue")
lines(fitted(garch.model) + residuals(garch.model), col="red")



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
