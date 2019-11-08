import numpy as np
import pandas as pd
from env import host, user, password

# Data import
url = f'mysql+pymysql://{user}:{password}@{host}/zillow'

def acquire_zillow():
    zillow_data = pd.read_sql('''Select *
From properties_2017
JOIN svi_db.`svi2016_us_county` as svi
    on zillow.`properties_2017`.`fips` = svi_db.svi.fips
    
Join
(SELECT
p_17.parcelid,
logerror,
transactiondate
FROM predictions_2017 p_17
JOIN 
(SELECT
  parcelid, Max(transactiondate) as tdate
FROM
  predictions_2017
  
Group By parcelid )as sq1
ON (sq1.parcelid=p_17.parcelid and sq1.tdate = p_17.transactiondate )) sq2
USING (parcelid)
WHERE (latitude IS NOT NULL AND longitude IS NOT NULL)
AND properties_2017.propertylandusetypeid NOT IN (31, 47,246, 247, 248, 263, 265, 267, 290, 291)
And properties_2017.unitcnt <= 1
and (properties_2017.bathroomcnt > 0)
And (properties_2017.bedroomcnt > 0)
;''', url)
    return zillow_data
acquire_zillow()

df = acquire_zillow()