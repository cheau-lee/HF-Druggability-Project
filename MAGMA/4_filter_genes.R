library(data.table)
library(dplyr)

# Load the data
magma_results <- fread('MAGMA/magma_output/magma_results_2.txt.genes.out')
genes <- fread('MAGMA/magma_data/NCBI37.3.gene.loc')

# Rename columns
setnames(genes, c('V1', 'V6'), c('GENE', 'HGNC_GENE'))

# Select relevant columns
genes <- select(genes, GENE, HGNC_GENE)

# Merge the data frames
df <- merge(magma_results, genes, by='GENE', all.x=TRUE)

# Define thresholds for filtering
thresholds <- c(0.05, 0.01, 0.001, 0.05/18295, 5e-8)

# Function to filter by threshold, get unique genes, and save results
filter_save_unique_genes <- function(data, threshold) {
  filtered_df <- filter(data, P < threshold)
  unique_genes <- unique(filtered_df[, .(GENE, HGNC_GENE)])
  # Save results to a file
  output_file <- paste0("unique_genes_threshold_", gsub("\\.", "", format(threshold, scientific = FALSE)), ".txt")
  write.table(unique_genes, file = output_file, row.names = FALSE, col.names = TRUE, quote = FALSE)
  return(unique_genes)
}

# Apply the function for each threshold and save results
for (threshold in thresholds) {
  unique_genes <- filter_save_unique_genes(df, threshold)
  cat("Threshold:", threshold, "\n")
  cat("Number of unique genes:", nrow(unique_genes), "\n")
  print(unique_genes)
  cat("\n")
