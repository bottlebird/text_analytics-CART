library(dplyr)

data = read.csv("returns.csv")
returns = data[,3:122]

head(data)
# Problem 2a

grep("avg200801",colnames(data))
grep("avg201012",colnames(data))

summary(data$Industry)
industry = aggregate(.~Industry, data=data, mean)[-2]
plot(t(industry[1,][25:60]), type="l", ylim=c(-0.3, 0.3), 
     xlab = "Month", ylab="Average Return %")
title("Consumer Discretionary")

industry = aggregate(.~Industry, data=data, mean)[-2]
plot(t(industry[2,][25:60]), type="l", ylim=c(-0.3, 0.3), 
     xlab = "Month", ylab="Average Return %")
title("Consumer Staples")

industry = aggregate(.~Industry, data=data, mean)[-2]
plot(t(industry[3,][25:60]), type="l", ylim=c(-0.3, 0.3), 
     xlab = "Month", ylab="Average Return %")
title("Energy")

industry = aggregate(.~Industry, data=data, mean)[-2]
plot(t(industry[4,][25:60]), type="l", ylim=c(-0.3, 0.3), 
     xlab = "Month", ylab="Average Return %")
title("Financials")

industry = aggregate(.~Industry, data=data, mean)[-2]
plot(t(industry[5,][25:60]), type="l", ylim=c(-0.3, 0.3), 
     xlab = "Month", ylab="Average Return %")
title("Health Care")

industry = aggregate(.~Industry, data=data, mean)[-2]
plot(t(industry[6,][25:60]), type="l", ylim=c(-0.3, 0.3), 
     xlab = "Month", ylab="Average Return %")
title("Industrials")

industry = aggregate(.~Industry, data=data, mean)[-2]
plot(t(industry[7,][25:60]), type="l", ylim=c(-0.3, 0.3), 
     xlab = "Month", ylab="Average Return %")
title("Information Technology")

industry = aggregate(.~Industry, data=data, mean)[-2]
plot(t(industry[8,][25:60]), type="l", ylim=c(-0.3, 0.3), 
     xlab = "Month", ylab="Average Return %")
title("Materials")

industry = aggregate(.~Industry, data=data, mean)[-2]
plot(t(industry[9,][25:60]), type="l", ylim=c(-0.3, 0.3), 
     xlab = "Month", ylab="Average Return %")
title("Telecommunications Services")

industry = aggregate(.~Industry, data=data, mean)[-2]
plot(t(industry[10,][25:60]), type="l", ylim=c(-0.3, 0.3), 
     xlab = "Month", ylab="Average Return %")
title("Utilities")

# Problem 2b
d = dist(returns)
mod.hclust = hclust(d, method="ward.D2") # Hierarchical Clustering
plot(mod.hclust, labels=F, xlab="", ylab="Dissimilarity", sub="") # Dendrogram

dissim.hc = data.frame(k=seq_along(mod.hclust$height), dissimilarity=rev(mod.hclust$height))
plot(dissim.hc$k, dissim.hc$dissimilarity, type="l", xlim=c(0,30), 
     xlab="Number of Clusters", ylab="Dissimilarity")

# Problem 2c
h.clusters = cutree(mod.hclust, 7)
h.cluster.result = data.frame(h.clusters)
data.h = data.frame(data$Industry)
data.h$cluster = as.factor(h.cluster.result$h.clusters)
data.h
result = aggregate(returns, by=list(h.clusters), mean) %>% select(-Group.1)
result
table(h.clusters)


# Problem 2d

km = kmeans(returns, centers=7, iter.max=100)

km.clusters = km$cluster
km.cluster.result = data.frame(km.clusters)
data.km = data.frame(data$Industry)
data.km$cluster = as.factor(km.cluster.result$km.clusters)
data.km

names(km)

km.centroids = km$centers

km$tot.withinss
km.size = km$size
km.size
table(h.clusters, km.clusters)


