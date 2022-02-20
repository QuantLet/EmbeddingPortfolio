rm(list = ls(all = TRUE))
graphics.off()

# install and load packages
# libraries = c("fGarch", "FinTS", "forecast", "xts")
# lapply(libraries, function(x) if (!(x %in% installed.packages())) {
#   install.packages(x)
# })
# lapply(libraries, library, quietly = TRUE, character.only = TRUE)


source("utils.R")
# library(fGarch)
# library(xts)

libraries = c("rjson", "forecast", "parallel", "doParallel")
lapply(libraries, function(x) if (!(x %in% installed.packages())) {
  install.packages(x)
})
lapply(libraries, library, quietly = TRUE, character.only = TRUE)

config = fromJSON(file="./config/dataset1.json")
run = config$run
data_specs = config$data_specs

# for loop here
cv = 1
data_spec = data_specs[[cv]]
# load dataset
data = get_cv_data("./data/dataset1/train_linear_activation.csv", 
                   data_spec$start, 
                   data_spec$val_start, 
                   data_spec$end,
                   config$window_size)
factors = colnames(data$train)
# for (factor.name in factors) {
#   print(factor.name)
# }
for (ind in 1:length(factors)){
  print(ind)
  factor.name = factors[ind]
  print(factor.name)
}
ind = 2
factor.name = factors[ind]
train_data = data$train[,ind]
val_data = data$val[,ind]

# Normalize data to have variance 1
train_data.sd = sd(train_data)
train_data = train_data / train_data.sd
val_data = val_data / train_data.sd

# Find best model on train set
best_model = model_selection(train_data, config$model.params, fit_model)

probas = predict_proba(train_data, val_data, config$window_size, best_model$model, 
                       fit_model, next_proba)
plot(probas, type="l")