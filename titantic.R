library(tidyverse)
library(readxl)
library(stringr)
library(skimr)

setwd("C:/R-projects/kaggle")

#Get a list of files in the input folder
input_path <- "input/"
output_path <- "output/"
process_path <- "processed/"

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