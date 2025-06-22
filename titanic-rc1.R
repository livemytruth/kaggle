library(tidyverse)
library(GGally)
library(ggplot2)
library(caret)
library(regclass)
library(cowplot)
library(gam)
library(reshape2)
library(xgboost)
library(Matrix)
library(randomForest)
library(rpart)
library(mice)
library(e1071)
library(gbm)
library(neuralnet)
library(skimr)


# 0. Set the environment --------------------------------------------------


setwd("C:/dataprojects/kaggle/kaggle")
set.seed(2)

#Get a list of files in the input folder
input_path <- "input/"
output_path <- "output/"
process_path <- "processed/"


# 1. Load the data --------------------------------------------------------


# Read in the CSV files
csv_list <- list.files(path = "input", pattern = "\\.csv$", full.name = TRUE, ignore.case = TRUE)

#Create an empty list to store data frames
file_list <- list()

#Loop through the list of files names and read in each CSV file into a DF
for(file in csv_list){
  variable_name <- sub(" ", "", sub("\\.csv$", "\\1", tools::file_path_sans_ext(basename(file))))
  if(grepl("_", deparse(substitute(variable_name)))){
    variable_name <- sub("^(.*?)_.*$", "\\1", variable_name)
  }
  assign(variable_name, read_csv(file))
  file_list[[variable_name]] <- get(variable_name)
}

alldata <- `titanic-test` %>% 
  mutate(Survived = rep(NA, 418))  %>%
  rbind(`titanic-train`)
  
# 2. Inspect the data --------------------------------

# Look for NAs by summing the number of NAs for each column
colSums(is.na(alldata)) # Missing values: Age [263], Fare [1], Cabin [1014], Embarked [2]

# Get the percentages of the missing values
missingValues <- alldata %>% 
  mutate(AgeNA = sum(is.na(Age))/n()*100,
         FareNA = sum(is.na(Fare))/n()*100,
         CabinNA = sum(is.na(Cabin))/n()*100,
         EmbarkedNA = sum(is.na(Embarked))/n()*100) %>% 
  select(contains("NA"), -Name) %>% 
  unique()
# In order of the most missing: Cabin, Age, Embarked, Fare


# 3. Handle Missing Values ------------------------------------------------

# 3.1 Handle Missing Embarked ---------------------------------------------

alldata %>% filter(is.na(Embarked))
# There are two records that are in teh same cabin, so it is probably a passenger and her caretaker
# Do a Google search to find out where they embarked from: Southampton, so set that manually

alldata <- alldata %>% 
  mutate(Embarked = ifelse(is.na(Embarked), "S", Embarked))


# 3.2 Handle Missing Fares ------------------------------------------------

alldata %>% filter(is.na(Fare)) #There is only one record who is 3rd class, age of 60.5, and no siblings/spouse
# since there is only one, we can impute from the average based on other features

# Need to determine if there is a correlation between age and fare
ggplot(data = subset(alldata, !(is.na(Age) | is.na(Fare)))) +
  geom_point(aes(x = Age, y = Fare)) +
  labs(x = "Age", y = "Fare")
# There is no correlation between age and fare, so we will impute values based on Sibsp, Parch, Embarked, and Pclass
impute_val <- alldata %>% 
  filter(SibSp == 0 & Parch == 0 & Embarked == "S" & Pclass == 3) %>% #filter for the matching values
  summarise(avgFare = mean(Fare, na.rm = TRUE)) #calculate the fare
alldata <- alldata %>% 
  mutate(Fare = ifelse(is.na(Fare), impute_val$avgFare, Fare))

# 3.2a Age Imputed as Averages --------------------------------------------

# Look at the correlation of Age with the other values
abs(cor(alldata[c("Age", "Pclass", "SibSp", "Fare", "Survived", "PassengerId", "Parch")], use = "complete.obs"))
# Most correlates with Pclass [0.3692] and then Survived [0.03579]

alldata_no_age_na <- alldata %>% drop_na(Age)
alldata2 <- alldata %>% 
  group_by(Pclass) %>% 
  mutate(Age = ifelse(is.na(Age), median(Age, na.rm = TRUE), Age))


# Appendix A: Helper functions --------------------------------------------

# Plotting histograms
plot_hist <- function(df, feature, cts, bin_size){
  if(cts == 1){
    p <- df %>% 
      group_by(var = cut())
  }
}
  
