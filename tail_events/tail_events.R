rm(list = ls(all = TRUE))
graphics.off()

# library("rjson")
source("utils.R")

TEST = TRUE
set_threshold=TRUE
dir_ = "prediction_w4months_q1%_q5%_q2%_q10%"

# ECON parameters
# save dir
save_dir = "./saved_models_Haindorf2020_results/"

# Respect order !!
qs = c(0.01, 0.025, 0.05, 0.1)
window_size = 4 * MONTH # Fit model on 4 months history
q_fit = 0.05  # fit to 10% of worst outcomes

set.seed(7)

dset.name  = paste(save_dir, dir_, "/dfdata.csv", sep = "")
dataset <- read.csv(dset.name, header = TRUE)

dates = dataset$X
rownames(dataset) = dates
dataset = dataset[ , -which(names(dataset) %in% c("returns","X"))]
dataset <- timeSeries::as.timeSeries(dataset, FinCenter = "GMT")

# Convert price series to losses and lower (negative) threshold to a positive threshold
dataset = cbind(dataset, c(NaN, - diff(log(dataset$close))))
colnames(dataset) = c(colnames(dataset)[1:ncol(dataset)-1], "loss")
head(dataset)


dataset[, lower_cols] = - dataset[, lower_cols]

# Start just before validation set
dataset = dataset[rownames(dataset) >= '2017-06-01 00:00:00', ]
length.dataset = nrow(dataset)
dates = rownames(dataset)
# Split data
test_size = length.dataset - window_size
prediction = matrix(nrow = test_size, ncol = (3 + (11 * length(qs))))

if (TEST){
    test_size = 20
}

count = 1
time <- Sys.time()
save_path = gsub(' ', '', gsub('-', '', gsub(':', '', time)))

save_path = paste0(save_dir, dir_, '/', save_path, sep="")
print(save_path)

########################

for (i in (window_size + 1):(window_size + test_size)) {
    if ((test_size + window_size - i) %% 100 == 0){
        print(paste0("Steps to go: ", test_size + window_size - i))
    }
    if (i %% 1000 == 0) {
        print(paste("Saving prediction at step", i))
        write.csv(prediction, paste0(save_path, "prediction_online_first.csv"), row.names = FALSE)
    }
    data.series = dataset[(i - window_size):(i - 1), "loss"]
    # Normalize entire dataset to have variance one
    return.sd = apply(data.series, 2, sd)
    data.series = data.series$loss / return.sd
    next.return = dataset[i, "loss"]
    lowers = dataset[i - 1, lower_cols]/ return.sd
    date = dates[i]

    # Fit model and get prediction
    if (!TEST){
      prediction_i =   tryCatch(
        fit_predParallel(date, data.series, next.return, return.sd, lowers, n=window_size, q_fit=q_fit,
                         qs=qs),
        error = function(e)
          base::rep(NA, ncol(prediction)), 
        silent = FALSE)
    }
    if (TEST){
      prediction_i =  fit_predParallel(date, data.series, next.return, return.sd, lowers, n=window_size, q_fit=q_fit,
                     qs=qs)
    }
    
    prediction[count,] = as.vector(prediction_i)
    count = count + 1
}

data_pred = prediction[!is.na(prediction[, 1]),]
dates = data_pred[, 1]
data_pred = data_pred[,-1]
mode(data_pred) = "double"
df = data.frame(data_pred)


cn = c()
for (q in qs){
    cn = c(cn,
    paste0("lower_", q),
    paste0("threshold_", q),
    paste0("var_evt_", q),
    paste0("es_evt_", q),
    paste0("proba_evt_", q),
    paste0("var_norm_", q),
    paste0("es_norm_", q),
    paste0("proba_norm_", q),
    paste0("mean_", q),
    paste0("sd_", q),
    paste0("zq_", q)
    )
}

colnames(df) = c(
"std_losses",
"norm_sd",
cn
)
rownames(df) = dates

######### SAVE
if (TEST){
    write.csv(df, paste0(
    paste0('./',
    save_path),
    "_prediction_10per_proba_TEST.csv"
    ),
    row.names = TRUE)
} else {
    write.csv(df,
    paste0(save_path,
    "_multi_q_prediction_qfit_",
    substr(as.character(q_fit),
    3,
    4),
    ".csv"),
    row.names = TRUE
    )
}
