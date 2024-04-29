###--------------------------------------------------------------------------###
###         COMM 301 | Machine Learning for Communication Management         ###
###                            Project Group 10                              ###
###--------------------------------------------------------------------------###

## Dependencies ====
load("COMM301_Project_G10.RData")

pacman::p_load(tidyverse, # Tidy DS
               tidymodels, # Tidy ML
               Hmisc, skimr, broom, jtools, huxtable, # EDA
               GGally, gridExtra, scales, ggthemes,
               DT, plotly, # interactive data display
               vip, # variable importance plot
               usemodels, ranger, # for computational engine for ML
               doParallel, # for speedy computation
               factoextra, tidyclust, cluster,
               themis, yardstick, pROC,
               workflows, workflowsets, doMC,
               recipes, interactions
)

setwd("~/Desktop/comm301/Project")

## Import Data ====

turnover <- 
  read_csv("https://talktoroh.squarespace.com/s/turnover.csv")

turnover %>% 
  glimpse()

## Question 1 ====
### 1.1. Tidy & Transform ----

turnover %>% 
  map_dbl(.,
          function(x) 
          {
            sum(is.na(x = x)
            )
          }
  )

A1a <-
  turnover %>% 
  mutate_if(is.character, as.factor) %>% 
  mutate(.data = .,
         Turnover = fct_relevel(Turnover,
                         "Yes")
         )

A1a %>% 
  skim()

### 1.2. Correlation matrix ----

corr_matrix_prep <-
  recipe(formula = Turnover ~ .,
         data = A1a) %>% 
  step_upsample(Turnover) %>% 
  step_normalize(all_numeric_predictors()
  ) %>% 
  step_poly(all_numeric_predictors(),
            role = "predictors",
            degree = 2) %>%
  step_dummy(all_nominal_predictors()
  ) %>% 
  prep(verbose = T) %>% 
  juice()

A1b <- 
  corr_matrix_prep %>% 
  mutate(Turnover = as.numeric(Turnover)
  ) %>% 
  as.matrix(.) %>%
  rcorr(.) %>%
  tidy(.) %>%
  mutate(absCorr = abs(estimate)
  ) %>%
  select(column1, column2, absCorr) %>% 
  datatable(.) %>% 
  formatRound(columns = c("absCorr"),
              digits = 3)

### 1.3. Split ----

set.seed(240417)

turnover_SPLIT <-
  A1a %>% 
  initial_split(prop = 0.80,
                strata = Turnover)

turnover_TRAIN <- 
  turnover_SPLIT %>% 
  training()

turnover_TEST <- 
  turnover_SPLIT %>% 
  testing()

turnover_TRAIN %>% 
  datatable()

## Question 2 ====
### 2.1. Checking Interactions ----

ckeck_interactions <-
  glm(formula = Turnover ~ (.):(.),
      data = turnover_TRAIN,
      family = binomial(link = "logit")
      )

ckeck_interactions_NEW %>%
  export_summs()

### 2.2. Pre-processing ----
#### 2.2.1. Upsample ----

A2_upsample <-
  recipe(formula = Turnover ~ .,
         data = turnover_TRAIN) %>%
  step_rm(NumCompaniesWorked) %>% 
  step_zv(all_predictors()
          ) %>% 
  step_upsample(Turnover) %>% 
  step_YeoJohnson(all_numeric_predictors()
                  ) %>% 
  step_normalize(all_numeric_predictors()
                 ) %>% 
  step_dummy(all_nominal_predictors()
             ) 

A2_upsample %>% 
  prep(verbose = T) %>% 
  juice(.) %>% 
  skim()

#### 2.2.2. Downsample ----

A2_downsample <-
  recipe(formula = Turnover ~ .,
         data = turnover_TRAIN) %>%
  step_zv(all_predictors()
          ) %>%
  step_downsample(Turnover) %>% 
  step_YeoJohnson(all_numeric_predictors()
                  ) %>% 
  step_normalize(all_numeric_predictors()
                 ) %>% 
  step_dummy(all_nominal_predictors()
             ) 

A2_downsample %>% 
  prep(verbose = T) %>% 
  juice(.) %>% 
  skim()

#### 2.2.3. Rose ----

A2_rose <-
  recipe(formula = Turnover ~ .,
         data = turnover_TRAIN) %>%
  step_zv(all_predictors()
          ) %>% 
  step_YeoJohnson(all_numeric_predictors()
                  ) %>% 
  step_normalize(all_numeric_predictors()
                 ) %>% 
  step_dummy(all_nominal_predictors()
             ) %>% 
  step_rose(Turnover) 

A2_rose %>% 
  prep(verbose = T) %>% 
  juice(.) %>% 
  skim()

#### 2.2.4. step_log() ----

A2_log <- 
  recipe(formula = Turnover ~ .,
       data = turnover_TRAIN) %>%
  step_log(DistanceFromHome, PercentSalaryHike) %>% 
  step_normalize(all_numeric_predictors()
                 ) %>% 
  step_dummy(all_nominal_predictors()
             ) %>% 
  step_corr(all_predictors(),
            threshold = 0.90)

A2_log %>% 
  prep(verbose = T) %>% 
  juice(.) %>% 
  skim()

#### 2.2.5. step_boxcox() ----

A2_boxcox <- 
  recipe(formula = Turnover ~ .,
         data = turnover_TRAIN) %>%
  step_BoxCox(DistanceFromHome, PercentSalaryHike) %>% 
  step_normalize(all_numeric_predictors()
                 ) %>% 
  step_dummy(all_nominal_predictors()
             ) %>%
  step_corr(all_predictors(),
            threshold = 0.90)

A2_boxcox %>% 
  prep(verbose = T) %>% 
  juice(.) %>% 
  skim()

#### 2.2.6. step_original() ----

A2_original <- 
  recipe(formula = Turnover ~ .,
         data = turnover_TRAIN) %>%
  step_normalize(all_numeric_predictors()
                 ) %>% 
  step_dummy(all_nominal_predictors()
             )

A2_original %>% 
  prep(verbose = T) %>% 
  juice(.) %>% 
  skim()

#### 2.2.7. step_poly() ----

A2_poly <- 
  recipe(formula = Turnover ~ .,
       data = turnover_TRAIN) %>%
  step_normalize(all_numeric_predictors()
                 ) %>% 
  step_dummy(all_nominal_predictors()
             ) %>% 
  step_poly(StockOptionLevel,
            degree = 2,
            role = "predictor") %>% 
  step_corr(all_predictors(),
            threshold = 0.90)

A2_poly %>% 
  prep(verbose = T) %>% 
  juice(.) %>% 
  skim()

#### 2.2.8. MaritalStatus ----

A2_ms <-
  recipe(formula = Turnover ~ MaritalStatus + StockOptionLevel,
         data = turnover_TRAIN) %>% 
  step_zv(all_predictors()
  ) %>% 
  step_upsample(Turnover) %>%
  step_normalize(all_numeric_predictors()
  ) %>% 
  step_dummy(all_nominal_predictors()
  ) %>% 
  step_corr(all_predictors(),
            threshold = 0.90) %>% 
  step_poly(StockOptionLevel,
            degree = 2,
            role = "predictor")

A2_ms %>% 
  prep(verbose = TRUE) %>%
  juice() %>%
  skim()

#### 2.2.9. OverTime ----

A2_overtime <-
  recipe(formula = Turnover ~ OverTime + StockOptionLevel,
         data = turnover_TRAIN) %>% 
  step_zv(all_predictors()
          ) %>% 
  step_upsample(Turnover) %>%
  step_normalize(all_numeric_predictors()
                 ) %>% 
  step_dummy(all_nominal_predictors()
             ) %>% 
  step_corr(all_predictors(),
            threshold = 0.90) %>% 
  step_poly(StockOptionLevel,
            degree = 2,
            role = "predictor")

A2_overtime %>% 
  prep(verbose = TRUE) %>%
  juice() %>%
  skim()

#### 2.2.10. MonthlyIncome ----

A2_income <-
  recipe(formula = Turnover ~ MonthlyIncome + StockOptionLevel,
         data = turnover_TRAIN) %>% 
  step_zv(all_predictors()
          ) %>% 
  step_upsample(Turnover) %>%
  step_normalize(all_numeric_predictors()
                 ) %>% 
  step_dummy(all_nominal_predictors()
             ) %>% 
  step_corr(all_predictors(),
            threshold = 0.90) %>% 
  step_poly(StockOptionLevel,
            degree = 2,
            role = "predictor")

A2_overtime %>% 
  prep(verbose = TRUE) %>%
  juice() %>%
  skim()

#### NEW A7 ----

A7_recipe <- 
  recipe(formula = Turnover ~ MonthlyIncome + TotalWorkingYears +
           Age + BusinessTravel + YearsAtCompany + DailyRate +
           MonthlyRate + HourlyRate + DistanceFromHome +
           YearsWithCurrManager,
         data = turnover_TRAIN) %>%
  step_upsample(Turnover) %>%
  step_normalize(all_numeric_predictors()
                 ) %>% 
  step_dummy(all_nominal_predictors()
             ) %>% 
  step_corr(all_predictors(),
            threshold = 0.90) 

A7_recipe %>% 
  prep(verbose = T) %>% 
  juice(.) %>% 
  skim()

## Question 3 ====
### 3.1. Fit ----

rf <- 
  rand_forest() %>% 
  set_args(mtry = tune(), 
           trees = 1000L
           ) %>% 
  set_engine("ranger",
             importance = "impurity") %>% 
  set_mode("classification")

grid_rf <- 
  expand.grid(mtry = c(3, 4, 5)
              )

### 3.2. Tuning ----
#### 3.2.1. Set cross-validation sets ----

cv10 <-
  turnover_TRAIN %>% 
  vfold_cv(v = 10,
           strata = Turnover)

cv10

#### 3.2.2. Define your grid control ----

grid_CONTROL <-
  control_grid(parallel_over = "everything",
               save_workflow = T,
               save_pred = T)

grid_CONTROL

#### 3.2.3. Define workflow`sets` ----

our_first_workflow_sets <-
  workflowsets::workflow_set(
    preproc = list(up = A2_upsample,
                   down = A2_downsample,
                   ro = A2_rose, 
                   log = A2_log,
                   box = A2_boxcox, 
                   ori = A2_original, 
                   poly = A2_poly,
                   ms = A2_ms,
                   ot = A2_overtime,
                   income = A2_income),
    models = list(randomforest = rf) 
  )

#### 3.2.4. Define tuning parameters ----

set.seed(24041701)

workflow_sets_PARAM <- 
  our_first_workflow_sets %>% 
  extract_workflow(x = .,
                   id = "ro_randomforest") %>% 
  extract_parameter_set_dials(.) %>% 
  update(mtry = mtry(range = c(1,
                               ncol(turnover_TRAIN)
                               )
                     )
         )

workflow_sets_PARAM

#### 3.2.5. Define performance metrics ----

metrics_for_tuning <-
  metric_set(accuracy,
             roc_auc,
             f_meas)

#### 3.2.6. Set paralell processing -----

registerDoParallel()

## for multi-core processing

max_cores <- 
  detectCores()

max_cores

registerDoMC(cores = max_cores)

#### 3.2.7. Build control tower of many models ----

control_tower_ML <-
  our_first_workflow_sets %>% 
  workflow_map(
    seed = 24041701,
    grid = 25,
    resamples = cv10, 
    control = grid_CONTROL, 
    param_info = workflow_sets_PARAM, 
    metrics = metrics_for_tuning, 
    verbose = T
  )

#### 3.2.8. Rank Your Models within TRAINING SET ----

A3 <-
  control_tower_ML %>% 
  rank_results(select_best = T,
               rank_metric = "roc_auc") %>% 
  pivot_wider(id_cols = c(wflow_id, rank, model),
              names_from = .metric,
              values_from = mean
  )

### 3.3 Finalize Model ----
#### 3.3.1. Get best parameters ----

best_model <-
  control_tower_ML %>% 
  extract_workflow_set_result(id = "box_randomforest") %>% 
  select_best(metric = "roc_auc")

#### 3.3.2. Fit final model ----

fit_final <- 
  control_tower_ML %>% 
  extract_workflow("box_randomforest") %>% 
  finalize_workflow(best_model) %>% 
  last_fit(turnover_SPLIT)

#### 3.3.3. Prediction ----

predictions_rf <-
  fit_final %>%
  collect_predictions()

## Question 4 ====
### 4.1. Confusion Matrix ----

A4 <-
  predictions_rf %>% 
  conf_mat(truth = Turnover, 
           estimate = .pred_class) %>% 
  pluck(1) %>% 
  as_tibble() %>%
  mutate(cm_colors = ifelse(Truth == "Yes" & Prediction == "Yes", "True Positive",
                            ifelse(Truth == "Yes" & Prediction == "No", "False Negative",
                                   ifelse(Truth == "No" & Prediction == "Yes", "False Positive",
                                          "True Negative")
                                   )
                            )
         ) %>% 
  ggplot(aes(x = Prediction, y = Truth)) + 
  geom_tile(aes(fill = cm_colors)) +
  scale_fill_manual(values = c("True Positive" = "green1",
                               "True Negative" = "green3",
                               "False Positive" = "tomato1",
                               "False Negative" = "tomato3")
  ) + 
  geom_text(aes(label = n), color = "white", size = 10) + 
  geom_label(aes(label = cm_colors), 
             vjust = 2) +
  labs(title = "Confusion Matrix") + 
  theme_fivethirtyeight() + 
  theme(axis.title = element_text(),
        legend.position = "none")

## Question 5 ====
### 5.1. Fit other models ----
#### 5.1.1. upsample ----

up_model <-
  control_tower_ML %>% 
  extract_workflow_set_result(id = "up_randomforest") %>% 
  select_best(metric = "roc_auc")

fit_upsample <- 
  control_tower_ML %>% 
  extract_workflow("up_randomforest") %>% 
  finalize_workflow(up_model) %>% 
  last_fit(turnover_SPLIT)

Pred_Truth_up <- 
  fit_upsample %>% 
  collect_predictions() %>% 
  mutate(algorithm = "Random Forest (upsample)")

#### 5.1.2. downsample ----

down_model <-
  control_tower_ML %>% 
  extract_workflow_set_result(id = "down_randomforest") %>% 
  select_best(metric = "roc_auc")

fit_downsample <- 
  control_tower_ML %>% 
  extract_workflow("down_randomforest") %>% 
  finalize_workflow(down_model) %>% 
  last_fit(turnover_SPLIT)

Pred_Truth_down <- 
  fit_downsample %>% 
  collect_predictions() %>% 
  mutate(algorithm = "Random Forest (downsample)")

#### 5.1.3. rose ----

ro_model <-
  control_tower_ML %>% 
  extract_workflow_set_result(id = "ro_randomforest") %>% 
  select_best(metric = "roc_auc")

fit_rose <- 
  control_tower_ML %>% 
  extract_workflow("ro_randomforest") %>% 
  finalize_workflow(ro_model) %>% 
  last_fit(turnover_SPLIT)

Pred_Truth_ro <- 
  fit_rose %>% 
  collect_predictions() %>% 
  mutate(algorithm = "Random Forest (rose)")

#### 5.1.4. log ----

log_model <-
  control_tower_ML %>% 
  extract_workflow_set_result(id = "log_randomforest") %>% 
  select_best(metric = "roc_auc")

fit_log <- 
  control_tower_ML %>% 
  extract_workflow("log_randomforest") %>% 
  finalize_workflow(log_model) %>% 
  last_fit(turnover_SPLIT)

Pred_Truth_log <- 
  fit_log %>% 
  collect_predictions() %>% 
  mutate(algorithm = "Random Forest (log)")

#### 5.1.5. boxcox ----

box_model <-
  control_tower_ML %>% 
  extract_workflow_set_result(id = "box_randomforest") %>% 
  select_best(metric = "roc_auc")

fit_box <- 
  control_tower_ML %>% 
  extract_workflow("box_randomforest") %>% 
  finalize_workflow(box_model) %>% 
  last_fit(turnover_SPLIT)

Pred_Truth_box <- 
  fit_box %>% 
  collect_predictions() %>% 
  mutate(algorithm = "Random Forest (box)")

#### 5.1.6. original ----

ori_model <-
  control_tower_ML %>% 
  extract_workflow_set_result(id = "ori_randomforest") %>% 
  select_best(metric = "roc_auc")

fit_ori <- 
  control_tower_ML %>% 
  extract_workflow("ori_randomforest") %>% 
  finalize_workflow(ori_model) %>% 
  last_fit(turnover_SPLIT)

Pred_Truth_ori <- 
  fit_ori %>% 
  collect_predictions() %>% 
  mutate(algorithm = "Random Forest (ori)")

#### 5.1.7. ms ----

ms_model <-
  control_tower_ML %>% 
  extract_workflow_set_result(id = "ms_randomforest") %>% 
  select_best(metric = "roc_auc")

fit_ms <- 
  control_tower_ML %>% 
  extract_workflow("ms_randomforest") %>% 
  finalize_workflow(ms_model) %>% 
  last_fit(turnover_SPLIT)

Pred_Truth_ms <- 
  fit_ms %>% 
  collect_predictions() %>% 
  mutate(algorithm = "Random Forest (ms)")

#### 5.1.8. ot ----

ot_model <-
  control_tower_ML %>% 
  extract_workflow_set_result(id = "ot_randomforest") %>% 
  select_best(metric = "roc_auc")

fit_ot <- 
  control_tower_ML %>% 
  extract_workflow("ot_randomforest") %>% 
  finalize_workflow(ot_model) %>% 
  last_fit(turnover_SPLIT)

Pred_Truth_ot <- 
  fit_ot %>% 
  collect_predictions() %>% 
  mutate(algorithm = "Random Forest (ot)")

#### 5.1.9. income ----

income_model <-
  control_tower_ML %>% 
  extract_workflow_set_result(id = "income_randomforest") %>% 
  select_best(metric = "roc_auc")

fit_income <- 
  control_tower_ML %>% 
  extract_workflow("income_randomforest") %>% 
  finalize_workflow(income_model) %>% 
  last_fit(turnover_SPLIT)

Pred_Truth_income <- 
  fit_income %>% 
  collect_predictions() %>% 
  mutate(algorithm = "Random Forest (income)")

#### 5.1.10. poly ----

poly_model <-
  control_tower_ML %>% 
  extract_workflow_set_result(id = "poly_randomforest") %>% 
  select_best(metric = "roc_auc")

fit_poly <- 
  control_tower_ML %>% 
  extract_workflow("poly_randomforest") %>% 
  finalize_workflow(poly_model) %>% 
  last_fit(turnover_SPLIT)

Pred_Truth_poly <- 
  fit_poly %>% 
  collect_predictions() %>% 
  mutate(algorithm = "Random Forest (poly)")

compare_predictions <- 
  bind_rows(Pred_Truth_box, Pred_Truth_down, Pred_Truth_income,
            Pred_Truth_log, Pred_Truth_ori, Pred_Truth_ot, 
            Pred_Truth_ro, Pred_Truth_ms, Pred_Truth_up,
            Pred_Truth_poly)

### 5.2. ROC-AUC Curve ----

A5 <-
  compare_predictions %>% 
  group_by(algorithm) %>% 
  roc_curve(Turnover,
            .pred_Yes) %>% 
  autoplot() +
  theme(legend.position = "right",
        text = element_text(size = 7)
        ) +
  labs(title= "Comparisons of Predictive Power between 10 Random Forest Models",
       subtitle= "Random Forest Poly performs better")

## Question 6 ====
### 6.1. Feature Importance ----

Feature_Importance <- 
  fit_final %>% 
  extract_fit_parsnip() %>% 
  vip(aesthetics = list(fill = "tomato3",
                        alpha = 0.40)
  ) +
  theme_fivethirtyeight() 

A6 <- Feature_Importance

## Question 7 ====

rf_A7 <-
  workflow() %>%
  add_model(rf) %>%
  add_recipe(A7_recipe)

set.seed(24041701)

tuned_A7 <-
  rf_A7 %>%
  tune_grid(resamples = cv10, 
            grid = grid_rf,
            metrics = metrics_for_tuning)

parameterstuned_A7 <-
  tuned_A7 %>%
  select_best(metric = "roc_auc")

finalized_workflow_A7 <-
  rf_A7 %>%
  finalize_workflow(parameterstuned_A7)

fit_A7 <-
  finalized_workflow_A7 %>%
  last_fit(turnover_SPLIT)

performance_A7 <- 
  fit_A7 %>%
  collect_metrics()

predictions_A7 <- 
  fit_A7 %>%
  collect_predictions()

Note <-
  paste("Revised algorithm does not show improvement over the",
        "previous optimal one developed. This is because many features",
        "are important. Including all is better than 10 identified.")

A7_renamed <- 
  performance_A7 %>%
  pivot_wider(names_from = .metric,
              values_from = .estimate) %>% 
  mutate(wflow_id = "A7 preformance")

A7 <-
  A3 %>%
  select(wflow_id, accuracy, roc_auc) %>%
  bind_rows(A7_renamed) %>% 
  select(wflow_id, accuracy, roc_auc)

## Save RAM Space into Hard Disk ====

save.image("COMM301_Project_G10.RData")
