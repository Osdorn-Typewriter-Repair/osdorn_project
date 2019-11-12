# Osdorn Typewriting Repair
## Analysis of Zillow log error data

### Goal 
Improve our original estimate of the log error by using clustering methodologies.


### Team 
Padraic Doran and Sean Oslin


### Included with Github
- Jupyter notebook with calculations.
- README with background on the project
- Text files with functions for the various stages of the product.
- Shapefile Folder with all requisite files for shapefle used.
- CSV of query results from MYSQL.

### Data source and SQL query
- Only data from 2017 were used. 

- Only single-unit properties (almost entirely single family homes) were analyzed.

- All properties had to have a minimum of one unit, one bathroom and one bedroom. 

- If a property was bought more than once during a year, the query was limited to the latest transaction.

- Records missing latitude and longitude were excluded because they are fundamental to the scope of the project. 

- 55,720 records were imported into Python


### Data preparation
- Started with 78 columns of data.

- Deleted duplicate columns, new total = 68

- Deleted attributes with less than 90% of cells filled, new total = 29

- Deleted rows with less than 75% of cells filled, total columns unchanged

- Deleted unneeded and redundant attributes, new total 14

- As the number of missing values was fairly small (no more than a few hundred), we decided to use linear regression to impute mising values for 'lotsizesquarefeet' and 'taxamount'

- Again,as the number of missing values was fairly small, we decided to use simple imputer (set for median) to calculate missing values for 'calculatedfinishedsquarefeet', 'landtaxvaluedollarcnt', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt'

- Deleted rows missing 'yearbuilt' as we could not think of a good method to impute year, new total 45,631

- All data types at this point were int64 and float. Converted all but one float (logerror) to ints. 

- Calculated outliers, using 4x the IQR as the upper and lower thresholds for defining outliers.  After this step, we eliminated all rows with logerror, lotsizesquarefeet, bathroomcnt, calculatedfinishedsquarefeet, calculatedfinishedsquarefeet, structuretaxvaluedollarcnt, taxvaluedollarcnt, and landtaxvaluedollarcnt upper outliers. The number of lower outliers were small, so we decided not to delete any rows, new total 38,500.

- Split data using train/test ratio of 70/30.

- Encoded 'bedroomcnt' and 'bathroom count' into 9 and 7 new variables, respectively. We contemplating grouping the higher numbers of beds and baths into variables indicating 5 or more. Because of time constraints we did not do this. We kept the original bed and bath variables for now. 

- Because of the nature of the data we considered both MinMax and Standard scalers. We settled on Standard as it is the most frequently used. Additionally, we chose to scale only the variables that with units of dollars and square feet.

### Data exploration
- All random states used 123.

- Our first set of clusters used K Means for the count of bedrooms and bathrooms.

- The Elbow Method determined that the opitimal number clusters (k) was 5.

- The size of the clusters varied from 1,300 to 9,000.

- We ran multiple cominations of variables to graphical search for clusters. Only the x-axis/y-axis combination of 'calculatedfinishedsquarefeet' and 'logerror' appeared to show clusters. 

- We wanted to check if there was a statistical difference between the mean 'logerror' for each cluster using a one-way ANOVA. To do this we needed to check if the distributions for each cluster were normally distributed. First, we checked graphically using distplot and Q-Q plots. The distplots looked fairly normal. The results of Q-Q plots were ambiguous. We next ran two statistical tests (Anderson-Darling and Shapiro-Wilks) to check for normality. The results for both tests indicated that the distributions were not normally distributed, even after we removed outliers. Accordingly, we did not run the planned ANOVA.

- When we ran a cluster analysis using the x-axis/y-axis combination of latitude and longitude, we discovered that our aggressive steps to remove outliers and missing values had completely eliminating Ventura and Orange counties. Only Los Angeles country remained. This would indicate that recording keeping and outliers are more prevalent in these two counties that border LA.
- Our second set of clusters used K Means for the count of latitude and longitude.

- The Elbow Method determined that the opitimal number clusters (k) was 4.

- The size of the clusters varied from 2,300 to 9,600.

- We ran multiple cominations of variables to graphical search for clusters. No obvious clusters appeared.

- Histograms of all the variables yielded some interesting (but not unexpected) results. First, the most common configuration of single unit residence is three bedroom and two bathrooms. Additionally, the property values and square feet skewed right (i.e. higher prices and larger lot/building sizes). Finally, the decades immediately following World War II were the heyday for home construction - tying with the expansion of the aerospace and entertainment industries and the Baby Boom. 

- Examining jointplots and correlations for select variables indicates that lots  have increased in size and have increased faster than the size of the residence. Also, while the number of number of bedrooms have increased over time, the number of bathrooms has increased faster. Finally, the value of the land is nearly perfectly correlated with the tax value (r = .95). The value of the structure on the land is almost inconsequential (r = .065). 

- We weren't sure if we were doing the cluster analysis correctly, so we decided to rerun with different parameters to see if we could get different answers and gain more experience with clustering. Given the maxim of "Location, location, location," we decide to to focus on the geographic data. 

- The new clusters did not appear to provide additional insight. Further, clusters do not seem to provide much insight without domain knowledge

### Modeling
- Did not do additional feature engineering. 

- Besides the baseline, we ran a standard linear regression, decision tree regression and a lasso cross validation. The results were only a slight improvement over the baseline. 

- We decided against applying our models to the test data because of the already poor performance of the models.

### RUN IT BACK

- We decided to see if we could apply some domain knowledge and conduct feature engineering and pre clustering clustering with locations by using bounding boxes.

- The bounding boxes themselves are not real tight, so there are gaps where there otherwise should not be. This will be corrected in future versions of the study.

- To run the Run It Back file, the user will need to install several package using Pip:
                                Geopandas, Descartes, and Shapely

- Addtionally, the shapefile is the streetmap taken from the United States Census Bureau website. I cannot gaurantee that a shapefile from another website will mesh as nicely as the one from the Census Bureau website. 

- Keep all of the files found in the shapefile folder!!! Those files are necessary for the streetmap to function correctly.

- The address used in the query for the shapefile is the absolute path where the file exists on the user's computer.

- After over removing outliers in the baseline run, the decision was made to not remove outliers. This would help preserve the number of samples in each region, which could be heavily influenced by those outliers. Uniform scaling was performed instead. 

- Maggie Guist can write some code. Like, damn.

- The cluster created for both the baseline and the second iteration was based more on the notion of location that it was related to any features about the house. Posterity might say that this was a mistake, but with a evaluating the notion that location was a prime driver of housing prices, and therefore error, was influetial in cluster design. The next iteration of modeling will and should include identifying features about the actual houses themselves AS WELL AS the already designed clusters.

- The technical details for each type of modeling conducted are located in the file. Random State = 123. All other features are considered to be default unless otherwise indicated in the notebook. 

- 
