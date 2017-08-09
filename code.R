# K-means clustering analysis
# k-means is a unsupervised machine learning clustering algorithm
# you feed it your desired number of clusters (k) and the algorithm determines 
# the grouping of data that has the minimum summed within-group error based off a distance caluclation
# below is brief example on data cleaning, pre-processing, prediction, interpretation and visualization
####################################################################
# Overview
## intro to the packages
## about the dataset
## cleaning the dataset + feature engineering
## preprocessing the data
## skree plot to determine the number of clusters
## the KCCA data class in R
## cluster assignments
## interpretating the results (cluster characteristics)
####################################################################
####################################################################
## intro to the packages
library(RODBC)
# RODBC isn't used below, but its a package to access a database and extract data
library(tidyr)
library(dplyr)
library(ggplot2)
library(readr)
library(stringr)
# ^ tidy verse packages. used in essentially every project. very handy
library(data.table)
# good for handling large datasets. fread() function reads in csv very fast.
library(Hmisc)
# helps convert R into HTML code. used for R presentations
library(lubridate)
# helps manage data and time variables
library(caret)
# a very useful machine learning package. data-splitting, data pre-processing, feature selection, model tuning. Great for K-means
library(flexclust)
# main function is as.kcca(). this generates a "k-centroids cluster analysis" supports distance measures and centroid computation
library(DT)
# the datatable() function in DT makes scrollable verions of dataframes. Very handy for R presentations

####################################################################
####################################################################
## about the dataset
# - pbr means "Polling Ballot Reader"
# - 980 is the election code for the 2016 general election
# - this dataset can be viewed as the log files of the ballot reader

pbr980 <- fread("") 
# save the file I sent you here ^
# this data set is imported mostly clean
# We did some data manipulation in the RODBC connection code
## the only important thing was that we filtered for a specifc event type (cast ballots)

conprec <- fread("")
# the consolidated precincts file goes here^
# precinct-level data that we pull in

####################################################################
####################################################################
## cleaning the dataset + feature engineering
pbr980 <- pbr980 %>% mutate(dt = as.Date(dt)) %>% mutate(day = weekdays(dt)) %>% 
  filter(day == "Tuesday") %>% 
  mutate(period=factor(period, levels=c("morning","lunch","afternoon","evening"), ordered=T)) %>%
  select(-day)

# this code reformats the dt (datetime) variable in the imported data
# in EDA, I saw that a small percentage of ballots were cast on Monday and Wednesday
# these are invlaid votes, (most likely machine tests)
# we also need to make period an ordered factor (on import, it was just a character variable I believe)

conprec980 <- filter(conprec, election_id == 980) %>%
  distinct(SerialNumber, .keep_all = T) %>%
  select(1, 5, 7)
# we filter it for data that is relevant to this election


pbr980join <- left_join(pbr980, conprec980, by = "SerialNumber") %>%
  filter(complete.cases(Precinct)) %>%
  select(12, 13, 6:10) %>%
  mutate(h7 = ifelse(hour == 7, 1, 0)) %>%
  mutate(h8 = ifelse(hour == 8, 1, 0)) %>%
  mutate(h9 = ifelse(hour == 9, 1, 0)) %>%
  mutate(h10 = ifelse(hour == 10, 1, 0)) %>%
  mutate(h11 = ifelse(hour == 11, 1, 0)) %>%
  mutate(h12 = ifelse(hour == 12, 1, 0)) %>%
  mutate(h13 = ifelse(hour == 13, 1, 0)) %>%
  mutate(h14 = ifelse(hour == 14, 1, 0)) %>%
  mutate(h15 = ifelse(hour == 15, 1, 0)) %>%
  mutate(h16 = ifelse(hour == 16, 1, 0)) %>%
  mutate(h17 = ifelse(hour == 17, 1, 0)) %>%
  mutate(h18 = ifelse(hour == 18, 1, 0)) %>%
  mutate(h19 = ifelse(hour == 19, 1, 0)) %>%
  mutate(h20 = ifelse(hour == 20, 1, 0)) %>%
  mutate(h21 = ifelse(hour == 21, 1, 0)) %>%
  select(-3)

# here, we are joining data from the consolidated precincts to the election data
# all the mutate() calls are assigning binary values. 
## we are breaking the voting day into 1 hour chunks and assigning
## 1 if the vote was cast in this hour, 0 if the vote was not cast in this hour
# ^ here we are still at the individual vote cast level

pbr980count <- group_by(pbr980join, Precinct) %>% count()
# here, we are grabbing the total votes cast for each precinct
# this is because we think that size / volume is a feature worth including in our cluster analysis

pbr980comp <- group_by( pbr980join, Precinct) %>%
  summarise_all( funs(mean)) %>%                   
  left_join( pbr980count, by = "Precinct") %>%
  select(-2)
# Important step!!!
## we are aggregating data at the precinct level
## the value in each cell is the poercentage of total votes cast in that time block
## you'll notice that morning + lunch + afternoon + evening = 1.00 = sum(h7:h10) for all the rows
## very important to get data in this format before running the analysis


####################################################################
####################################################################
## preprocessing the data
# data pre-processing: centering, scaling, bringing back in the precict variable

#First, let's center and scale this data, as number of ballots will likely skew the clustering.
library(caret)
preproc980 <- preProcess(pbr980comp[,-1], method = c("center", "scale"))
# the [,-1] ^ here is beacuase we don't want to center and scale the precinct code (our 1st column)

pbr980trans <- predict(preproc980, pbr980comp[,-1])

pbr980trans$Precinct <- pbr980comp$Precinct

# because we used summarise_all (from above)
# the "n" column was preprocessed by accident. Below, we will bring back in the original units (not centered and scaled)
 
pbr980trans$n <- pbr980comp$n

pbr980transfin <- select(pbr980trans, 21, 1:20)



####################################################################
####################################################################
## skree plot to determine the number of clusters
# skree plot that shows the sum of squared difference within each cluster
NumClusters <- seq(2,10,1)
SumWithinss <- sapply(2:10, function(x) sum(kmeans(pbr980transfin[,2:21], 
                                                   centers = x, iter.max=1000)$withinss))
df_980 <- data.frame(NumClusters, SumWithinss)
ggplot(df_980, aes(x=NumClusters, y = SumWithinss)) + geom_point(size = 1.5, color = "pink") + geom_line()

# in this plot, we have the number of clusters on the x and a summed difference / error metric on the y
# when we see a steep drop from x to x + 1, this means that k = x + 1 clusters sharply reduces the summed error
# it stands to reason in this caser that x + 1 is a better number of clusters (more ordered) than x
# note that error is monotonically decreasing, so we are looking for that correct balance of small error, and small number of clusters
# also consider the number of variables or features that are in your model when choosing k

####################################################################
####################################################################
## the KCCA data class in R
# now that we've preprocess the data, and have found a good number of clusters to try, we are ready to run k-means
library(flexclust)
set.seed(112358)
km7 <- kmeans(pbr980transfin[,2:21], centers = 7)  

km.kcca7 = as.kcca(km7, pbr980transfin[,2:21])
save(km.kcca7, file = "km.kccapbr980.RData") 
# IMPORTANT to convert to kcca dataclass and save it
# just one of the things you have to do


####################################################################
####################################################################
## cluster assignments
## interpretating the results (cluster characteristics)
#Let's now get the cluster assignments
clusters = predict(km.kcca7)

pbr980comp$clusters <- clusters
# these are the cluster assignments

# What does each cluster consist off?

pbr980prec <- group_by(pbr980comp, clusters )%>%
  count() %>%
  setnames(2, "precinclust")

pbr980clust <- group_by(pbr980comp, clusters) %>%
  summarise_at(2:21, mean)

datatable(pbr980prec) 
# this has the number of precincts in each cluster

pbr980clust <- pbr980clust %>% mutate_if(is.numeric, funs(round(. ,digits = 3)))
datatable(pbr980clust)  
# this has the profiles of each cluster :)


####################################################################
####################################################################
## interpretating the results (cluster characteristics)
t1 <- pbr980clust %>% dplyr::select(-c(6:21)) %>%
  gather("period", "value", 2:5) %>% 
  mutate(period=factor(period, levels=c("morning","lunch","afternoon","evening"), ordered=T))


ggplot(t1, aes(x = period , y = value, fill = period)) + geom_bar(stat = "identity") + 
  facet_wrap(~clusters) + theme(axis.title.x = element_blank(), axis.text.x=element_blank(), axis.ticks.x=element_blank())
# this visualizes the distirbution of voting for each cluster
# IMprotant to plot the group average and see how each cluster deviates from it
## Run the same code with k = 4 and see how a differen number of clusters produces different results
