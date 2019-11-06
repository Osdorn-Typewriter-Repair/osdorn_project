# osdorn_project

### Goal 
Improve our original estimate of the log error by using clustering methodologies.

### Team 
Padraic Doran and Sean Oslin


### Data source and SQL query
- All data originated with the Zillow database and svi_db (for county names).
- Only data from 2017 were used. 
- Only single-unit properties (almost entirely single family homes) were analyzed.
- All properties had to have a minimum of one unit, one bathroom and one bedroom. 
- If a property was bought more than once during a year, the query was limited to the latest transaction.
- Records missing latitude and longitude were excluded because they are fundamental to the scope of the project. 
- 55,720 records were imported into Python

### Data preparation
- Started with 78 columns of data.
- Deleted duplicate columns, total = 68
- Deleted attributes with less than 90% of cells filled, new total = 29
- Deleted rows with less than 75% of cells filled, total unchanged
- Deleted unneeded and redundant attributes, new total 14
- As the number of missing values was fairly small (no more than a few hundred), we decided to use linear regression to impute mising values for 'lotsizesquarefeet' and 'taxamount'
- Again,as the number of missing values was fairly small, we decided to use simple imputer (set for median) to calculate missing values for 'calculatedfinishedsquarefeet', 'landtaxvaluedollarcnt', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt'
- Deleted rows missing 'yearbuilt' as we could not think of a good method to impute year, new total 55,624
- All data types at this point were int64 and float. Converted all but one float (logerror) to ints. 
Outliers