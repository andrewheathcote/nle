library(arrow)
library(dplyr)
library(ggplot2)

# TODO change paths to the right data path or use subsample
# load data
var_names <- c("param_id", "C", "rt") # variables to be extracted
prior_datDDM <- open_dataset('data/prior/DDM/trials') |> # lazy reading
  select(all_of(var_names)) |> # select variables
  dplyr::collect() |> # load as tibble
  mutate(model = "DDM") # add model name

prior_datRDM <- open_dataset('data/prior/RDM/trials') |>
  select(all_of(var_names)) |>
  dplyr::collect() |>
  mutate(model = "RDM")

prior_datLNR <- open_dataset('data/prior/LNR/trials') |>
  select(all_of(var_names)) |>
  dplyr::collect() |>
  mutate(model = "LNR")

prior_dat <- bind_rows(prior_datDDM, prior_datRDM, prior_datLNR)


# Proportion correct across parameter sets
prior_dat |>
  group_by(param_id, model) |>
  summarise(prop_correct = mean(C)) |>
  ungroup() |>
  ggplot(aes(x = model, y = prop_correct, fill = model)) +
  geom_violin(position = 'dodge', alpha = 0.8) +
  labs(x = "Model", y = "Proportion Correct") +
  theme_minimal()

# numerical investigation into RDM
# there seem to be some combinations where you only get an incorrect response
prior_dat |>
  group_by(param_id, model) |>
  summarise(prop_correct = mean(C)) |>
  filter(model == "RDM") |>
  arrange(prop_correct)

open_dataset('data/prior/RDM/trials') |> collect() |> filter(param_id == 437)

# RT distributions
prior_dat$rt <- ifelse(prior_dat$C, prior_dat$rt, -prior_dat$rt) # sign RTs

prior_dat |>
  ggplot(aes(x = rt)) +
  geom_density(alpha = .7, adjust = 2, fill = "darkgrey", linewidth = .75) +
  facet_wrap(~model) +
  lims(x = c(-5, 5)) +
  labs(y = "Density", x = "RT", fill = "Model") +
  theme_minimal() +
  theme(strip.text = element_text(face = 'bold', size = 12))

