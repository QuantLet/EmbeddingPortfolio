rm(list = ls(all = TRUE))
graphics.off()
source("run.R")
library(rjson)

# ------------------------ Input ------------------------
# Input
dataset = "dataset2"
config_path = "./config/dataset2.json"
save = TRUE
save_dir = "output"
do.debug = FALSE


# ------------------------ Script ------------------------

config = fromJSON(file = config_path)
if (save) {
  cwd = getwd()
  save_dir = file.path(cwd, save_dir)
  if (!dir.exists(save_dir)) {
    dir.create(save_dir)
  }
  save_dir = file.path(save_dir, dataset)
  if (!dir.exists(save_dir)) {
    dir.create(save_dir)
  }
  save_dir = file.path(save_dir, config$run)
  if (!dir.exists(save_dir)) {
    dir.create(save_dir)
  }
  time <- Sys.time()
  save_dir = file.path(save_dir, gsub(' ', '', gsub('-', '', gsub(':', '', time))))
  if (!dir.exists(save_dir)) {
    dir.create(save_dir)
  }
  train_save_path = file.path(save_dir, "train_activation_probas.csv")
  save_path = file.path(save_dir, "activation_probas.csv")
} else {
  save_path = NULL
}

# Main loop
t1 = Sys.time()
result = run(config, save_dir = save_dir, debug = do.debug)
t2 = Sys.time()
print(paste("Total time:", t2 - t1))

# Finally
if (save) {
  write.zoo(result$train, file = train_save_path, sep = ",")
  write.zoo(result$test, file = save_path, sep = ",")
}