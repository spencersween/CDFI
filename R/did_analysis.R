# Spencer Sween's CDFI DiD Analysis

# Clear Enviornment
rm(list = ls())

# Clear Memory
gc()

# Load Libraries
library(DoubleML)
library(mlr3)
library(mlr3learners)
library(mlr3extralearners)
library(tidyverse)
library(did)
library(DescTools)
library(fixest)
library(fastglm)

# Set Seed
set.seed(42)

# Migrate to Main Project Directory
setwd("~/Dropbox/CDFI Project/")
setwd("Data/Final_Data/Test/")

################################################################################
########## Import Data ##########

filename = "cdfi_startup_zip_V2.csv"
df = data.table::fread(filename) %>% 
  as.data.frame()

################################################################################
########## Create Formula ##########

cluster = "Cluster_county"
text_X = df %>%
  select(all_of(cluster),
         X_total_pop,
         X_prop_hispanic,
         X_prop_black,
         X_prop_less_hs,
         X_prop_hs_somecoll,
         X_prop_18_29,
         X_prop_30_39,
         X_prop_never_married,
         X_prop_unemployed,
         X_prop_poverty,
  ) %>%
  colnames() %>%
  paste(collapse = "+")
options(expressions = 50000)
formula = as.formula(paste("~ -1 +", text_X))

##################################################################a##############
########## Choose Estimation Settings ##########

# # Play Around
# outcome = "Y_log_sfr"
# cohort = 1994
# lag = 10
# data = df %>% filter(X_prop_white < 0.69) %>%
#   filter(group == 0 | group == cohort) %>%
#   filter(time == (cohort - 1) | time == (cohort - 1 + lag)) %>%
#   select(id, time, group, cluster, outcome, starts_with("X_"))
# y0 = data %>% filter(time == (cohort - 1)) %>% select(outcome) %>% as.matrix()
# y1 = data %>% filter(time == (cohort - 1 + lag)) %>% select(outcome) %>% as.matrix()
# D = data %>% filter(time == (cohort - 1 + lag)) %>% mutate(D = as.numeric(group > 0)) %>% select(D) %>% as.matrix()
# covariates = model.matrix(formula, data %>% filter(time == (cohort - 1 + lag)))

# Doubly Robust ATT with OLS
dr_att_ols = function(y, post, D, covariates, ...) {
  
  # Set Cut Points
  winsor = 0
  ptrim  = 1e-12
  
  # Obtain Data from Unbalanced Panel
  y1 = Winsorize(y[post == 1], probs = c(0, 1-winsor))
  y0 = Winsorize(y[post == 0], probs = c(0, 1-winsor))
  D = as.numeric(D[post == 0] > 0)
  covariates = covariates[post == 1,]
  
  ###########################################################
  # 1. Prepare Data
  ###########################################################
  
  # Extract Variables
  target = as.numeric(y1 - y0)
  treatment = as.numeric(D)
  cluster = as.numeric(covariates[,1])
  features = as.matrix(covariates[,-c(1)])
  
  # New Variables
  y = target
  X = lapply(1:ncol(features), function(c) poly(features[,c], 3, raw = TRUE))
  X = do.call(cbind, X)
  X = apply(X, 2, function(c) (c-min(c))/(max(c)-min(c)))
  X = cbind(1, X)
  d = treatment
  
  ###########################################################
  # 3. Fit Regression Models
  ###########################################################
  
  # Obtain Conditional Mean Coefficients
  b0 = coef(lm(y ~ X - 1, subset = (d == 0)))
  m0x = as.numeric(X %*% b0)
  
  # Obtain Propensity Scores
  logit = fastglm(x = X, y = d, intercept = FALSE, family = binomial)
  px = logit$fitted.values
  px = pmin(1 - ptrim, px)
  
  # Obtain Unconditional Propensity Score
  q = mean(d)
  
  ###########################################################
  # 4. Construct Influence Functions
  ###########################################################
  
  psi_theta = (1/q) * (d - (1-d)*(px/(1-px))) * (y - m0x)
  psi_q = -1 * (1/q) * mean(psi_theta) * (d - q)
  att = mean(psi_theta)
  iff = (psi_theta - att) + psi_q
  
  ###########################################################
  # 4. Return Output
  ###########################################################
  
  # Print Results
  test_data = tibble(psi = att + iff, cluster = cluster)
  test_estimates = fixest::feols(psi ~ 1, test_data, cluster = ~cluster)
  print(test_estimates)
  
  # Store Results in List
  ATT = att
  att.inf.func = matrix(0, nrow = length(post))
  att.inf.func[post == 1,] = (iff) * (1/mean(post))
  output = list(ATT = ATT, att.inf.func = att.inf.func)
  
  # Return
  return(output)
  
}

# Doubly Robust ATT with DMl and LightGBM
dr_att_dml = function(y, post, D, covariates, ...) {
  
  # Set Seed and Supress Training Messages
  # lgr::get_logger("mlr3")$set_threshold("warn")
  set.seed(1234)
  
  # Set Cut Points
  winsor = 0
  ptrim  = 1e-12
  
  # Obtain Data from Unbalanced Panel
  y1 = Winsorize(y[post == 1], probs = c(0, 1-winsor))
  y0 = Winsorize(y[post == 0], probs = c(0, 1-winsor))
  D = as.numeric(D[post == 0] > 0)
  covariates = covariates[post == 1,]
  
  ###########################################################
  # 1. Prepare Data
  ###########################################################
  
  # Extract Variables
  target = as.numeric(y1 - y0)
  treatment = as.numeric(D)
  cluster = as.numeric(covariates[,1])
  features = as.matrix(covariates[,-c(1)])
  
  # New Variables
  y = target
  X = lapply(1:ncol(features), function(c) poly(features[,c], 1, raw = TRUE))
  X = do.call(cbind, X)
  X = apply(X, 2, function(c) (c-min(c))/(max(c)-min(c)))
  d = treatment
  
  ###########################################################
  # 3. Fit Regression Models
  ###########################################################
  
  # Train LightGBM
  ml_g = lrn("regr.lightgbm", learning_rate = 0.01, num_leaves = 100, max_depth = -1)
  ml_m = lrn("classif.lightgbm", learning_rate = 0.01, num_leaves = 100, max_depth = -1)
  dml_data = double_ml_data_from_matrix(X = X, y = y, d = d, cluster_vars = cluster)
  dml_obj = DoubleMLIRM$new(dml_data,
                            ml_g = ml_g,
                            ml_m = ml_m,
                            score = "ATTE",
                            apply_cross_fitting = TRUE,
                            n_folds = 2)
  dml_obj$fit(store_predictions = TRUE)
  
  # Obtain Conditional Mean
  m0x = as.numeric(dml_obj$predictions$ml_g0)
  
  # Obtain Propensity Scores
  px = as.numeric(dml_obj$predictions$ml_m)
  px = pmin(1 - ptrim, px)
  
  # Obtain Unconditional Propensity Score
  q = mean(d)
  
  ###########################################################
  # 4. Construct Influence Functions
  ###########################################################
  
  psi_theta = (1/q) * (d - (1-d)*(px/(1-px))) * (y - m0x)
  psi_q = -1 * (1/q) * mean(psi_theta) * (d - q)
  att = mean(psi_theta)
  iff = (psi_theta - att) + psi_q
  
  ###########################################################
  # 4. Return Output
  ###########################################################
  
  # Print Results
  test_data = tibble(psi = att + iff, cluster = cluster)
  test_estimates = fixest::feols(psi ~ 1, test_data, cluster = ~cluster)
  print(test_estimates)
  
  # Store Results in List
  ATT = att
  att.inf.func = matrix(0, nrow = length(post))
  att.inf.func[post == 1,] = (iff) * (1/mean(post))
  output = list(ATT = ATT, att.inf.func = att.inf.func)
  
  # Return
  return(output)
  
}

# ATTGT Estimator
get_attgt = function(outcome, method, data) {
  att_gt_ms = att_gt(yname = outcome, 
                     tname = "time", 
                     idname = "id", 
                     gname = "group", 
                     clustervars = "Cluster_county", 
                     xformla = formula,
                     est_method = method,
                     data = data, 
                     panel = TRUE, 
                     allow_unbalanced_panel = TRUE,
                     anticipation = 0, 
                     base_period = "varying",
                     control_group = "notyettreated",
                     alp = 0.05, 
                     bstrap = TRUE, 
                     cband = TRUE, 
                     biters = 999,
                     print_details = TRUE, 
                     pl = TRUE, 
                     cores = parallel::detectCores() - 1)
}

# Event Study Estimator
get_atte = function(attgt, type, pre, post, cluster) {
  att_event = aggte(attgt,
                    type = type,
                    min_e = pre, 
                    max_e = post, 
                    na.rm = TRUE,
                    bstrap = TRUE, 
                    biters = 999, 
                    cband = TRUE, 
                    alp = 0.05,
                    clustervars = cluster)
}

# Event Study Grapher
ggdid_plotter = function(es, title = "", lim_l = NULL, lim_u = NULL) {
  plot = ggdid(es, type = "dynamic", theming = FALSE, linewidth = 2) +
    geom_ribbon(aes(ymin = es$att.egt - es$crit.val.egt * es$se.egt, 
                    ymax = es$att.egt + es$crit.val.egt * es$se.egt, 
                    fill = factor(es$egt >= 0)),  # Ensure this is a factor for legend grouping
                alpha = 0.50) +
    geom_line(aes(y = es$att.egt), linewidth = 0.25, linetype = "solid", color = "black", alpha = 0.5, show.legend = FALSE) +  # Hide line legend
    geom_vline(xintercept = -0.5, linetype = "dashed", linewidth = 0.25, show.legend = FALSE) +  # Hide vline legend
    scale_y_continuous(n.breaks = 10, labels = scales::number_format(accuracy = 0.001)) +
    scale_color_viridis_d(option = "viridis", begin = 0.75, end = 0.25, guide = "none") +  # Remove color legend
    scale_fill_viridis_d(option = "viridis", begin = 0.75, end = 0.25, labels = c("Pre-treatment", "Post-treatment")) + # Customize fill legend
    theme_classic() +
    ylab("Point Estimate and 95% UCB") +
    xlab("Years since First CDFI Activity") +
    theme(legend.position = "bottom", legend.title = element_blank(), legend.text = element_text(size = 10)) + # Remove legend title
    ggtitle(label = title)
  
  if (!is.null(lim_l) & !is.null(lim_u)) {
    plot = plot + scale_y_continuous(n.breaks = 11, 
                                     breaks = seq(lim_l, lim_u, length.out = 11), 
                                     limits = c(lim_l, lim_u), 
                                     labels = scales::number_format(accuracy = 0.01))
  }
  
  # Remove geom_errorbar layer if it exists
  plot$layers <- plot$layers[!sapply(plot$layers, function(layer) inherits(layer$geom, "GeomErrorbar"))]
  
  
  return(plot)
}

##################################################################a##############
########## Get Main Results ##########

# Overall Effects
df_overall = df
attgt_overall = get_attgt("Y_log_sfr", dr_att_dml, data = df_overall)
event_overall = get_atte(attgt_overall, "dynamic",pre = -15, post = 14, cluster = "Cluster_county")
plot_overall = ggdid_plotter(event_overall, lim_l = -0.15, lim_u = 0.50)
event_overall
plot_overall

# High Minority Effects
df_high = df %>% filter(X_prop_white < 0.69)
attgt_high = get_attgt("Y_log_sfr", dr_att_dml, data = df_high)
event_high = get_atte(attgt_high, "dynamic", pre = -15, post = 14, cluster = "Cluster_county")
plot_high = ggdid_plotter(event_high, lim_l = -0.15, lim_u = 0.50)
event_high
plot_high

# Low Minority Effects
df_low = df %>% filter(X_prop_white > 0.96)
attgt_low = get_attgt("Y_log_sfr", dr_att_dml, data = df_low)
event_low = get_atte(attgt_low, "dynamic", pre = -15, post = 14, cluster = "Cluster_county")
plot_low = ggdid_plotter(event_low, lim_l = -0.15, lim_u = 0.50)
event_low
plot_low

