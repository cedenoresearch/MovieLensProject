if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Training set used will be 50% of edx data to train algorithm's
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.5, list = FALSE)
train <- edx[-test_index,]
temp <- edx[test_index,]

# Make sure userId and movieId in test set are also in train set
test <- temp %>% 
  semi_join(train, by = "movieId") %>%
  semi_join(train, by = "userId")

# Add rows removed from validation set back into train set
removed <- anti_join(temp, test)
train <- rbind(train, removed)

##################################################################################
###################### Methods & Analysis ########################################
##################################################################################

# Head
head(train)

# Glimpse of dataset
glimpse(train)

# Summary of unique movies and users
summary(train)

# Number of unique movies and users in the train dataset
train %>% 
  summarize(n_users = n_distinct(userId), 
            n_movies = n_distinct(movieId), 
            n_genres = n_distinct(genres)) 

# Ratings distribution 
train %>% 
  ggplot(aes(rating)) + 
  geom_histogram(binwidth = 0.25, color = "black") + 
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) + 
  scale_y_continuous(breaks = c(seq(0, 3000000, 500000))) + 
  ggtitle("Rating distribution") # highest rating given was a 4; lowest rating given was .5

# Plot number of ratings per movie
train %>%
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "black") +
  scale_x_log10() +
  xlab("Number of ratings") +
  ylab("Number of movies") +
  ggtitle("Number of ratings per movie") # ~300 movies rated ~50+ times; but there is an
                                         # outlier of movies that were rated only once?

# Table of 20 movies rated only once
train %>%
  group_by(movieId) %>%
  summarize(count = n()) %>%
  filter(count == 1) %>%
  left_join(train, by = "movieId") %>%
  group_by(title) %>%
  summarize(rating = rating, n_rating = count) %>%
  slice(1:20) %>%
  knitr::kable() 

# Plot number of ratings given by users
train %>%
  count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "black") +
  scale_x_log10() +
  xlab("Number of ratings") + 
  ylab("Number of users") +
  ggtitle("Number of ratings given by users") # removing the outliers of 20 movies rated only once

# Plot mean movie ratings given by users
train %>%
  group_by(userId) %>%
  filter(n() >= 100) %>%
  summarize(b_u = mean(rating)) %>%
  ggplot(aes(b_u)) +
  geom_histogram(bins = 30, color = "black") +
  xlab("Mean rating") +
  ylab("Number of users") +
  ggtitle("Mean movie ratings given by users") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  theme_light() # ~1850 users gave ratings ~3.5

##################################################################################
###################### Modeling Approaches ########################################
##################################################################################

# Calculate the average rating of all movies in the dataset
mu_hat <- mean(train$rating)
mu_hat # 3.512439 as estimated visually (see above plots)

# Test results based on simple prediction on the edx test set 
naive_rmse <- RMSE(test$rating, mu_hat)
naive_rmse # 1.060035

# Creating a results dataframe that contains all RMSE results
rmse_results <- data.frame(model="Naive Mean-Baseline Model", RMSE=naive_rmse)
rmse_results %>% knitr::kable()

## Movie Effect Model ##
# Calculate the average by movie: 
# Simple model taking into account the movie effect b_i
# Subtract the rating minus the mean for each rating the movie received
# Plot number of movies with the computed b_i
movie_avgs <- train %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu_hat))
movie_avgs %>% qplot(b_i, geom="histogram", bins = 10, data = ., color = I("black"), 
                     ylab = "Number of Movies", main = "Number of Movies with computed b_i")

# Test and save rmse results 
predicted_ratings <- mu_hat + test %>%
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)
model_1_rmse <- RMSE(predicted_ratings, test$rating)
rmse_results <- bind_rows(rmse_results,
                          data.frame(model="Movie effect model",  
                                     RMSE = model_1_rmse ))

# Check results
rmse_results %>% knitr::kable() # .9439087 RMSE (ok it's minimizing)


## Movie and user effect model ##
# Calculate the average by user:
# Plot the penalty term = user effect 
user_avgs <- train %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  filter(n() >= 100) %>% 
  summarize(b_u = mean(rating - mu_hat - b_i))
user_avgs %>% qplot(b_u, geom="histogram", bin=30, data=., color=I("black"), 
                    ylab = "Number of Movies", main = "Number of Movies with computed b_i and b_u")

user_avgs <- train %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_hat - b_i))

# Test and save rmse results 
predicted_ratings <- test %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu_hat + b_i + b_u) %>%
  pull(pred)

model_2_rmse <- RMSE(predicted_ratings, test$rating)
rmse_results <- bind_rows(rmse_results,
                          data.frame(model="Movie and user effect model",  
                                     RMSE = model_2_rmse))

# Check result
rmse_results %>% knitr::kable() # RMSE minimizes to 0.8653488 !

## Movie, user, and genre effect model ##

# Calculate the average genre popularity by user:
genre_pop <- train %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu_hat - b_i - b_u))

# Compute the predicted ratings on validation dataset
predicted_ratings <- test %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_pop, by='genres') %>%
  mutate(pred = mu_hat + b_i + b_u + b_g) %>%
  pull(pred)

model_3_rmse <- RMSE(predicted_ratings, test$rating)
rmse_results <- bind_rows(rmse_results,
                          data.frame(model="Movie, user, and genre effect model",  
                                     RMSE = model_3_rmse))

# Check result
rmse_results %>% knitr::kable() # RMSE minimizes to 0.8649469!


## Regularized movie, user, and genre effect model ##
# lambda is a tuning parameter
# Use cross-validation to choose it.
lambdas <- seq(0, 10, 0.25)

# For each lambda, find b_i, b_u, b_u_g, followed by rating prediction & testing using 
# the edx & validation (final-hold-out) data.
# note:the below code could take some time  
rmses <- sapply(lambdas, function(l){
  # Calculate the average of all ratings in 90% of data
  mu <- mean(edx$rating)
  # Add movie effect
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  # Add user effect
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  # Add genre effect
  b_g <- edx %>% 
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by='userId') %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+l))
  # Compute the predicted ratings on validation dataset (10% of data)
  predicted_ratings <- validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    mutate(pred = mu + b_i + b_u + b_g) %>%
    pull(pred)
  # Predict the RMSE on the validation set (10% of data)
  return(RMSE(predicted_ratings, validation$rating))
})

# Plot rmses vs lambdas to select the optimal lambda                                                            
qplot(lambdas, rmses)  

# The optimal lambda: the lambda value that minimizes the RMSE                                                             
lambda <- lambdas[which.min(rmses)]
lambda

# Test and save results                                                             
rmse_results <- bind_rows(rmse_results,
                          data_frame(model="Regularized movie, user, and genre effect model",  
                                     RMSE = min(rmses)))

# Check result
rmse_results %>% knitr::kable() 

#### Results ####                                                            
# RMSE results overview                                                          
rmse_results %>% knitr::kable() 
# The Regularized movie, user, and genre effect model yields
# a minimized RMSE of 0.8644501
