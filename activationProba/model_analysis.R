rm(list = ls(all = TRUE))
graphics.off()

# install and load packages
# libraries = c("fGarch", "FinTS", "forecast", "xts")
source("utils.R")
config_path = "./config/config.json"
config = fromJSON(file = config_path)

dataset = "dataset1"
window_size = 250
model_name = "20220301004321"
cv = 0
factor.name = "SPX_X"

# Load model
model_path = file.path("./output", dataset, model_name, cv, paste0(factor.name, "_model.rds"))
model <- readRDS(model_path)

# Load data
data = get_cv_data(dataset, cv, window_size = window_size)
print(paste0("Last train: ", index(tail(data$train, 1))[1]))
print(paste0("First test: ", index(data$test)[1]))
data = data$train
factors = colnames(data)
index = index(data)
ind = which(factors == factor.name)
fdata = data[,ind]
head(fdata)

# Plot data
plot(fdata, ylab=factor.name)
# graphics.off()
# dev.off()

# Plot ACF os Squared Standardized Residuals
par(mfrow = c(1, 1), bg="transparent")
plot(model, which=11)
png('ACF_res.png', bg = "transparent")

# QQ-Plot of Standardized Residuals
par(mfrow = c(1, 1), bg="transparent")
plot(model, which=13)

# To control plot param need to call qdist and .qqLine
model.residuals = xts(fGarch::residuals(model, standardize = TRUE), order.by=index(fdata))
zseries = as.numeric(model.residuals)
n = length(zseries)
if (model@fit$params$cond.dist == "sstd") {
  npar = length(model@fit$par)
  xi = model@fit$par[npar-1] # Skew
  nu = model@fit$par[npar] # Shape
  par(mfrow = c(1, 1), bg="transparent")
  plot(qsstd(ppoints(n)[order(order(zseries))], nu=nu, xi=xi), zseries, 
       ylab="Sample Quantiles", xlab="Theoretical Quantiles", xlim= c(-4, 4), ylim= c(-4, 4))
  qqline(zseries, datax = FALSE, distribution = qsstd, 
         main = expression("Q-Q plot for" ~~ {qsstd}[nu=nu, xi=xi]))
} else {
  stop("Only implemented for sstd")
}
png('qqplot_res.png', units="px", bg = "transparent")
