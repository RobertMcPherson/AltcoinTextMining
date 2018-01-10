#' Compiling several text files into a single CSV file
#' 
#' Convert a folder of text files into a single CSV file 
#' with one column for the file names and one column of the 
#' text of the file. A function in R.
#' 
#' To use this function for the first time run this next line:
#' install.packages("devtools")
#' then thereafter you just need to load the function 
#' fom github like so, with these two lines:
#' library(devtools) # windows users need Rtools installed, mac users need XCode installed
#' source_url("https://gist.github.com/benmarwick/9265414/raw/text2csv.R")
#' 
#' Here's how to set the arguments to the function:
#' 
#' mydir is the full path of the folder that contains your txt files
#' for example "C:/Downloads/mytextfiles" Note that it must have 
#' quote marks around it and forward slashes, which are not default
#' in windows.
#' 
#' mycsvfilename is the name that you want your CSV file to 
#' have, it must have quote marks around it, but not
#' the .csv bit at the end
#' 
#' A full example, assuming you've sourced the 
#' function from github already:
#'
#' txt2csv("C:/Downloads/mytextfiles", "mybigcsvfile")
#'
#' and after a moment you'll get a message in the R console
#' saying 'Your CSV file is called mybigcsvfile.csv and 
#' can be found in C:/Downloads/mytextfiles'


txt2csv <- function(mydir, mycsvfilename){
  
  # Get the names of all the txt files (and only txt files)
  myfiles <- list.files(mydir, full.names = TRUE, pattern = "*.txt")
  
  # Read the actual contexts of the text files into R and rearrange a little.
  
  # create a list of dataframes containing the text
  mytxts <- lapply(myfiles, read.delim)
  
  # combine the rows of each dataframe to make one
  # long character vector where each item in the vector
  # is a single text file
  mytxts1lines <- unlist(lapply(mytxts, function(i) unname(apply(i, 2, paste, collapse=" "))))
  
  # make a dataframe with the file names and texts
  mytxtsdf <- data.frame(filename = basename(myfiles), # just use filename as text identifier
                         fulltext = mytxts1lines) # full text character vectors in col 2
  
  # Now write them all into a single CSV file, one txt file per row
  
  setwd(mydir) # make sure the CSV goes into the dir where the txt files are
  # write the CSV file...
  write.table(mytxtsdf, file = paste0(mycsvfilename, ".csv"), sep = ",", row.names = FALSE, col.names = FALSE)
  # now check your folder to see the csv file
  message(paste0("Your CSV file is called ", paste0(mycsvfilename, ".csv"), ' and can be found in ', getwd()))
}
