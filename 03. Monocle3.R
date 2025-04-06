library(Seurat)
library(monocle3)
library(ggplot2)

expr_matrix <- pbmc[["RNA"]]$data  
meta_data <- pbmc@meta.data  
gene_annotation <- data.frame(gene_short_name = rownames(expr_matrix), row.names = rownames(expr_matrix))  


pbmc_cds <- new_cell_data_set(expr_matrix,
                              cell_metadata = meta_data,
                              gene_metadata = gene_annotation)

pca_results <- pbmc[["pca"]]@cell.embeddings

reducedDims(pbmc_cds)$PCA <- pca_results

umap_results <- pbmc[["umap"]]@cell.embeddings

reducedDims(pbmc_cds)$UMAP <- umap_results

pbmc_cds <- cluster_cells(pbmc_cds, reduction_method = "UMAP", resolution = 0.5)

pbmc_cds <- learn_graph(pbmc_cds)

pbmc_cds <- order_cells(pbmc_cds)

pseudotime <- pseudotime(pbmc_cds,reduction_method = 'UMAP')

umap_coords <- reducedDims(pbmc_cds)$UMAP

umap_plot <- ggplot() +
  geom_point(aes(x = umap_coords[, 1], y = umap_coords[, 2], color = pseudotime), size = 1) +
  scale_color_gradient(low = "blue", high = "red") +
  labs(x = "UMAP_1",
       y = "UMAP_2",
       color = "Pseudotime"
  ) +
  theme_minimal() +
  theme(legend.title = element_text(size = 20),
        axis.title.x = element_text(size = 20),
        axis.title.y = element_text(size = 20),
        legend.text = element_text(size = 18),
        axis.text.x = element_text(size = 18),
        axis.text.y = element_text(size = 18),
        panel.grid = element_blank()
  )
print(umap_plot)

umap_plot <- ggplot() +
  geom_point(aes(x = umap_coords[, 1], y = umap_coords[, 2], color = pseudotime), size = 1) +
  scale_color_gradient(low = "blue", high = "red") +
  theme_void() +  
  theme(legend.position = "none",  
        plot.background = element_rect(fill = "transparent", color = NA),  
        panel.background = element_rect(fill = "transparent", color = NA))  


ggsave(filename = "UMAP_with_Pseudotime.png", 
       plot = umap_plot, 
       width = 10, 
       height = 8, 
       units = "in", 
       dpi = 300)

