library(dplyr)
library(Seurat)
library(patchwork)
library(stringr)
library(ggplot2)
library(biomaRt)
library(dplyr)
library(Seurat)
library(ggplot2)
library(WGCNA)
options(stringsAsFactors = FALSE)
allowWGCNAThreads()

pbmc.data <- read.table("/GSE112013/GSE112013_Combined_UMI_table.txt", sep="\t", header=TRUE,row.names=1)
pbmc <- CreateSeuratObject(counts = pbmc.data, project = "pbmc")

pbmc <- NormalizeData(pbmc)

pbmc <- FindVariableFeatures(pbmc, selection.method = "vst", nfeatures = 2000)

all.genes <- rownames(pbmc)
pbmc <- ScaleData(pbmc, features = all.genes)

pbmc <- RunPCA(pbmc, features = VariableFeatures(object = pbmc))

DimPlot(pbmc, reduction = "pca") + NoLegend()
ElbowPlot(pbmc)

pbmc <- FindNeighbors(pbmc, dims = 1:20)
pbmc <- FindClusters(pbmc, resolution = 0.5)
table(Idents(pbmc))
dataIDs <- Idents(pbmc)

pbmc <- RunUMAP(pbmc, reduction = "pca", dims = 1:20)
classIDs<- Idents(pbmc)

Idents(pbmc) <- classIDs

exclude_celltypes <- c("2", "10", "9", "15", "8")

cells_to_keep <- rownames(pbmc@meta.data)[!(pbmc@meta.data$seurat_clusters %in% exclude_celltypes)]
pbmc_filtered <- pbmc[, cells_to_keep]

DefaultAssay(pbmc_filtered) <- "RNA"

human = useMart(biomart = "ENSEMBL_MART_ENSEMBL", dataset = "hsapiens_gene_ensembl",host = "http://jul2023.archive.ensembl.org/")
data <- read.table("/GSE109037/features.tsv", header = F, sep = "\t")

gene_id = getBM(attributes = c("ensembl_gene_id","gene_biotype"),
                filters = "ensembl_gene_id",
                values = data$V1,
                mart = human)

protein_coding_genes <- gene_id[gene_id$gene_biotype == "protein_coding", ]

ensembl_ids <- protein_coding_genes$ensembl_gene_id

unique_ensembl_ids <- unique(ensembl_ids)

filtered_data <- data[data$V1 %in% unique_ensembl_ids, ]

pbmc_filtered2 <- subset(pbmc_filtered, features = filtered_data$V2)

DefaultAssay(pbmc_filtered2) <- "RNA"
mat_filtered <- AggregateExpression(pbmc_filtered2, group.by = 'seurat_clusters', slot = "counts")
mat_filtered <- NormalizeData(mat_filtered$RNA)
mat_filtered_df <- as.data.frame(mat_filtered)
datExpr= as.data.frame(t(mat_filtered_df)); 
powers = c(c(1:10), seq(from = 12, to=20, by=2))
sft = pickSoftThreshold(datExpr, powerVector = powers, verbose = 5)
sft
sizeGrWindow(9, 5)
par(mfrow = c(1,2));
cex1 = 0.9;

pdf("./Scale-free.pdf")
plot(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2],
     xlab="Soft Threshold (power)",ylab="Scale Free Topology Model Fit,signed R^2",type="n",
     main = paste("Scale independence"));
text(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2],
     labels=powers,cex=cex1,col="red");
abline(h=0.9,col="red")
plot(sft$fitIndices[,1], sft$fitIndices[,5],
     xlab="Soft Threshold (power)",ylab="Mean Connectivity", type="n",
     main = paste("Mean connectivity"))
text(sft$fitIndices[,1], sft$fitIndices[,5], labels=powers, cex=cex1,col="red")
dev.off()

net <- blockwiseModules(datExpr, power = 18,
                        corType = "pearson", 
                        networkType = "signed", minModuleSize = 30,
                        reassignThreshold = 0, mergeCutHeight = 0.25,
                        numericLabels = T, pamRespectsDendro = FALSE,
                        saveTOMs = TRUE,
                        saveTOMFileBase = "TOM_all",
                        verbose = 3)


