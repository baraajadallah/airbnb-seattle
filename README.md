# airbnb-seattle

## Motivation and about the project

This project is part of the **Udacity** data science nano-degree program and it's mainly about two parts, this **Github** repository and a blog post. In this project, I am working on the Seattle **Airbnb** data to answer the following questions:

1- How does the number of listings differ between the different neighbourhoods in the city?

2- What are the most common amenities in general, and what are they in the most frequent neighbourhoods?

3- What are the properties that affects listing prices the most? In addition prices predictions of new listings.

## About the data

The data in this project is aquired from **Kaggle** website and I worked mainly on the Listings csv file. In order to answer my questions, I went through several steps which are deatiled in the code files.

## Findings

We notice that the most common amenities are almost similar between the general approach and in the top neighbourhoods, and that as I see is from two reasons, the first one is that the top 20 neighbourhoods listings numbers represent the vast majority of the total listings, and the second is that the most common amenities are so crusial amd almost no one can live without, such as kitchen, heating and wireless internet, however we notice some change in positions when the amenities become less common.

We figure out that few numerical variables have reasonable correlation with the prices, and as we represent this in the following pair plot, we notice the positive correlation between price and the other numerical features, except for the reviews per month variable, I think the negative correlation is due to booking less expensive listings by people and as a result cheape listings gain more number of reviews.

The best results we achieved is less than 48 dollars test RMSE error on average, and the model explains 67% of the variability in listing price, by RandomForestRegressor. The results are not the greatist bu believe more features are needed and those features should be sharp and direct.

And from the coefficiants data frame, we notice that the physical features of the rooms themselves are affecting the pricing more than other features such as the neighbourhood or low priority amenities.