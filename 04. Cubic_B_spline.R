library(dplyr)
library(splines)
library(tidyverse)

order <- c("g11","g5","g6","g13","g14","g12","g7","g4","g3","g1","g0")

mat_filtered_df <- mat_filtered_df[, order]

x <- seq(0, 1, length.out = 11) 

B <- bs(x, df = 7, degree = 3) 

Bcoef <- matrix(0, nrow(mat_filtered_df), 7)

for (i in 1:nrow(mat_filtered_df)) {
  y <- as.matrix(mat_filtered_df[i, ])  
  y <- matrix(y, ncol = 1)  
  Bcoef[i, ] <- solve(t(B) %*% B) %*% t(B) %*% y
}

Bcoef_df <- data.frame(Bcoef)

colnames(Bcoef_df) <- paste0("Bcoef_", 1:7)

Bcoef_df$genes <- rownames(mat_filtered_df)
