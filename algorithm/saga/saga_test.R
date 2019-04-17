
data(BreastCancer, package="mlbench")
bc <- BreastCancer[complete.cases(BreastCancer), ]

bc$Cell.size  <- as.numeric(bc$Cell.size)
bc$Cell.shape <- as.numeric(bc$Cell.shape)

m <- glm(Class ~ Cell.size + Cell.shape, family="binomial", data = bc[1:20,])
