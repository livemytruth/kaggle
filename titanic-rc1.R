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

# 3.3 Age Imputed as Regression Trees --------------------------------------------

alldata$Title <- substring(alldata$Name, regexpr(',', alldata$Name) + 2, regexpr('[.]', alldata$Name) - 1)
ageFit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare,  alldata)
alldata$Age[is.na(alldata$Age)] <- predict(ageFit, alldata[is.na(alldata$Age),])


# 3.3 Handle Missing Cabin Values -----------------------------------------
# The number of missing values are way too many to impute, so we cannot use the cabin feature.
# We can get some values from it, though.
# Looking at the data, the first letter of the cabin denotes the deck.
# Those with no cabin, put them in an imaginary deck.

alldata <- alldata %>% 
  mutate(Deck = ifelse(is.na(Cabin), "X", substr(Cabin, 1, 1)))
unique(alldata$Deck) # Values: A, B, C, D, E, F, G, T, X
alldata %>% group_by(Deck) %>% count()
alldata %>% filter(Deck == "T") #Look at the one record with T deck, and it is someone with a first class ticket, so putting it with the A group
alldata <- alldata %>% 
  mutate(Deck = ifelse(Deck == "T", "A", Deck))

# Let's look at the distribution
# Calculate the deck data
p <- alldata %>% 
  group_by(Deck) %>% 
  summarise(n_pass_per_Deck = n(),
            p_class_1 = sum(Pclass == 1)/n_pass_per_Deck,
            p_class_2 = sum(Pclass == 2)/n_pass_per_Deck,
            p_class_3 = sum(Pclass == 3)/n_pass_per_Deck,
            p_survived = sum(Survived, na.rm = TRUE)/n_pass_per_Deck,
            p_died = 1 - p_survived)

# Build two more dataframes translating from wide to long
p2 <- melt(p[, c('Deck', 'p_class_1', 'p_class_2', 'p_class_3')], is.vars = 1)
p3 <- melt(p[, c('Deck', 'p_survived', 'p_died')], is.vars = 1)

# Plot  the new dataframes, so we can see the distribution
plot_grid(
  ggplot(p2, aes(x = Deck, y = value)) +
    geom_bar(aes(fill = variable), stat = "identity", position = "dodge") +
    my_custom_theme +
    fill_values,
  ggplot(p3, aes(x = Deck, y = value)) +
    geom_bar(aes(fill = variable), stat = "identity", position = "stack") +
    my_custom_theme +
    fill_values,
  align = "v", rel_heights = c(2,2), nrow = 2)

# Analysis
# The trends are as expected; however, they is a very high mortality rate in the
# imaginary deck, so it cannot be used to predict whether someone dies.

# Look at the naive model that predicts that everyone beside those on deck X survived does much better than random guessing.
diedinX <- nrow(alldata[which(alldata$Survived == 0 & alldata$Deck == "X"),]) #Count the number that did not survive on X
livedoutsideX <- nrow(alldata[which(alldata$Survived == 1 & alldata$Deck != "X"),])
print(paste("Naive model accuracy:", 100*(diedinX + livedoutsideX)/891, "%")) #At 70%

# Get rid of Cabin
alldata <- alldata %>% select(-Cabin)


# 4: Explore the Data -----------------------------------------------------


# 4.1: Age (Continuous) vs. Survivability ---------------------------------
# This is a continuous variable, so it has to be binned.
plot_hist(alldata, "Age", 1, 8)
plot_surv(alldata[1:891,], "Age", 1, 8)

# Age has some good split points for a decision tree. 


# 4.2: Fare (Continuous) vs. Survivability --------------------------------

# It is continuous, so it needs to be binned.
plot_hist(alldata, "Fare", 1, 24)
plot_surv(alldata[1:891,], "Fare", 1, 24)



# Appendix A: Helper functions --------------------------------------------

# Plotting histograms
plot_hist <- function(df, feature, cts, bin_size){
  if(cts == 1){
    p <- df %>% 
      group_by(var = cut(!!sym(feature), breaks=seq(0, max(!!sym(feature)), bin_size))) %>% 
      summarise(num = n())
    ggplot(p, aes(x = var, y = num, fill = factor(var))) + geom_bar(stat = "identity") +
      theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
            plot.title = element_text(size = 10, face = "bold")) +
      xlab(paste(feature, "bins")) +
      ylab("Value Count") +
      ggtitle(paste("Passengers by", feature), ) +
      theme(legend.position = "none") +
      my_custom_theme +
      fill_values
  } else {
    p <- df %>% 
      group_by(var = !!sym(feature)) %>% 
      summarise(num = n())
    ggplot(p, aes(x = var, y = num, fill = factor(var))) + geom_bar(stat = "identity") +
      theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
            plot.title = element_text(size = 10, face = "bold")) +
      xlab(paste(feature, "bins")) +
      ylab("Value Count") +
      ggtitle(paste("Passengers by", feature), ) +
      theme(legend.position = "none") +
      my_custom_theme +
      fill_values
  }
}

plot_surv <- function(df, feature, cts, bin_size){
  if(cts == 1){
    p <- df %>% 
      group_by(var = cut(!!sym(feature), breaks = seq(0, max(!!sym(feature)), bin_size))) %>% 
      summarise(num = n(),
                n_survived = sum(Survived, na.rm = TRUE),
                n_died = num - n_survived,
                p_survived = n_survived/num,
                p_died = 1-p_survived)
    p2 <- melt(p[,c('var', 'n_survived', 'n_died', 'p_survived', 'p_died')], id.vars = 1)
    plot_grid(
      ggplot(p2[1:floor(nrow(p2)/2),], aes(x = var, y = value)) +
        geom_bar(aes(fill = variable), stat = "identity", position = "dodge") +
        theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
              legend.posistion = "top",
              legend.title = element_blank(),
              plot.title = element_text(size = 10, face = "bold")) +
        xlab(paste(feature, "bins")) +
        ylab("Value counts") +
        ggtitle(paste("Number of survivors by", feature)) +
        scale_fill_discrete(labels = c("Survived", "Perished")) +
        my_custom_theme + fill_values,
      ggplot(p2[floor(1 + nrow(p2)/2):nrow(p2),], aes(x = var, y = value)) +
        geom_bar(aes(fill = variable), stat = "identity", position = "stack") +
        theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
              legend.posistion = "top",
              legend.title = element_blank(),
              plot.title = element_text(size = 10, face = "bold")) +
        xlab(paste(feature, "bins")) +
        ylab("Proportions") +
        ggtitle(paste("Proportion of survivors by", feature)) +
        scale_fill_discrete(labels = c("Survived", "Perished")) +
        my_custom_theme + fill_values
    )
    
  } else {
    p <- df %>% 
      group_by(var = !!sym(feature)) %>% 
      summarise(num = n(),
                n_survived = sum(Survived, na.rm = TRUE),
                n_died = num - n_survived,
                p_survived = n_survived/num,
                p_died = 1-p_survived)
    ggplot(p, aes(x = var, y = num, fill = factor(var))) + geom_bar(stat = "identity") +
      theme(legend.position = "none") + xlab(paste(feature, "bins")) + ylab("value counts")
    p2 <- melt(p[,c('var', 'n_survived', 'n_died', 'p_survived', 'p_died')], id.vars = 1)
    plot_grid(
      ggplot(p2[1:floor(nrow(p2)/2),], aes(x = var, y = value)) +
        geom_bar(aes(fill = variable), stat = "identity", position = "dodge") +
        theme(legend.posistion = "top",
              legend.title = element_blank(),
              plot.title = element_text(size = 10, face = "bold")) +
        xlab(paste(feature)) +
        ylab("Value counts") +
        ggtitle(paste("Number of survivors by", feature)) +
        scale_fill_discrete(labels = c("Survived", "Perished")) +
        my_custom_theme + fill_values,
      ggplot(p2[floor(1 + nrow(p2)/2):nrow(p2),], aes(x = var, y = value)) +
        geom_bar(aes(fill = variable), stat = "identity", position = "stack") +
        theme(legend.posistion = "top",
              legend.title = element_blank(),
              plot.title = element_text(size = 10, face = "bold")) +
        xlab(paste(feature)) +
        ylab("Proportions") +
        ggtitle(paste("Proportion of survivors by", feature)) +
        scale_fill_discrete(labels = c("Survived", "Perished")) +
        my_custom_theme + fill_values
    )
  }
}


# Appendix B: Set the Theme -----------------------------------------------

# Define your custom theme
my_custom_theme <- theme_minimal() +
  theme(
    # Plot title
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5, color = "#4B616F"),
    
    # Axis titles
    axis.title = element_text(size = 12, face = "bold", , color = "#4B616F"),
    
    # Axis text
    axis.text = element_text(size = 10, color = "#9E978E"),
    
    # Legend
    legend.title = element_text(size = 12, face = "bold" , color = "#4B616F"),
    legend.text = element_text(size = 10, color = "#9E978E"),
    legend.position = "bottom",
    
    # Grid lines
    panel.grid.major = element_line(color = "#B8C5CE"),
    panel.grid.minor = element_line(color = "#B8C5CE"),
    
    # Background
    panel.background = element_rect(fill = "#F3F1D8", color = "#4B616F", linewidth = 0.5, linetype = "solid"),
    plot.background = element_rect(fill = "#F3F1D8", color = "#4B616F", linewidth = 0.5, linetype = "solid"),
    
    # Margins
    plot.margin = margin(10, 10, 10, 10)
  )

fill_values <- scale_fill_manual(values = c("#CC2344", "#F2C462", "#8CBEB2", "#4B616F", "#603934",
                                            "#F06060", "#EA7434", "#1AB895", "#9E978E", "#B8C5CE"))
