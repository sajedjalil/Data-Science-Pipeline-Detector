import pandas as pd
# ## Doing this type of analysis is against the competition rules
# 
# It has been pointed out by the competition admin that incorporating external data about distances between cities is against the rules.
# 
# So please don't use this in any way for building your models!
# 
# 
# 
# ## The locations puzzle
# 
# Expedia presented us with a dataset where countries and cities are hidden behind integer codes. Is it possible to find out which city is which?
# 
# Let's grab our pandas and find out :).
# 
# Read in a few lines to get a list of columns.
train = pd.read_csv('../input/train.csv', nrows=10)
train.columns
# Columns related to user location are:
# 
# - `user_location_country`
# - `user_location_region`
# - `user_location_city`
# 
# Columns related to hotel location are:
# 
# - `hotel_country`
# - `hotel_market`
# - `srch_destination_id`
# 
# Finally, the `orig_destination_distance` column shows us the distance in miles between the user and their chosen hotel.
# 
# We should note that hotel countries and user location countries are encoded differently, meaning that the same country will have different numbers in these two columns. I will not go into `srch_destination_id`s yet because they might represent different locations within the same city so this division is probably too fine. On the other hand the `hotel_market`s correspond to nonoverlapping regions all over the globe, and large cities are covered by their own `hotel_market`s so this is a nice match to `user_location_city` column.
# 
# Now let's read in a million rows using only columns relating to this task. Drop rows where distance is undefined.
train = pd.read_csv('../input/train.csv', usecols = ['posa_continent', 
       'user_location_country',
       'user_location_region', 'user_location_city',
       'orig_destination_distance','hotel_continent', 
       'hotel_country', 'hotel_market'], nrows=1000000).dropna()
# ## A mapping of user and hotel countries
# 
# If a user books a hotel in their own city then we should see a very short distance in the corresponding dataset row. So we can look at which user and hotel countries have the lowest minimum distances between them and deduce that these pairs probably refer to the same actual country.
# 
# Let's group rows by user country and hotel country and look at the distances.
distaggs = (train.groupby(['user_location_country','hotel_country'])
            ['orig_destination_distance']
            .agg(['min','mean','max','count']))
distaggs.sort_values(by='min').head(20)
# So we see a huge number of rows belonging to user country 66 and hotel country 50. It's probably the USA. Then there are some more pairs with low distances.
# 
# First repeated row is user country 205 and hotel country 50 again. And the minimum distance is 3 miles here - larger than 0.0056 we saw in the first rows. So user country 205 must be some neighbor country, Canada or Mexico.
# 
# Then there's the repeat appearance of user country 66 with travels to hotel countries 8 and 198 - those are probably again Canada and Mexico. 
# 
# By the end of the table shown minimum distances go up to almost 45 miles, so this criterion does not work so obviously any more - the pairs might be just neighboring countries, and not necessarily the same one.
# 
# ## user_location_country 66
# 
# How many regions does this country have?
c66 = train[train.user_location_country==66]
c66.user_location_region.unique().shape
# 51 looks fitting for the USA.
# 
# Let's look at trips within this country.
# 
# The USA have Hawaii which is a popular tourist location and also very far away from other regions. I'll group the data by user_location_region and hotel_market and take a look at maximum distances.
c66in = c66[c66.hotel_country==50]
(c66in.groupby(['user_location_region','hotel_market'])['orig_destination_distance']
      .agg(['min','mean','max','count'])
      .sort_values(by='max',ascending=False).head(20))
# Looks like we have a lot of hotel_market values 212, 214 and a couple of 213 for good measure. These could be our paradise islands.
# 
# Let's look at distances from hotel_market 212 to different user cities in the USA sorting by popularity.
hawaii = (c66in[c66in.hotel_market == 212]
          .groupby(['user_location_region','user_location_city'])
          ['orig_destination_distance']
          .agg(['min','mean','max','count'])
          .sort_values(by='count',ascending=False))
hawaii.head(10)
# Looks like we caught some very local trips in row 4. So region 246 is probably Hawaii.
# 
# The site http://www.distancefromto.net/ tells me that distances from Honolulu to other cities are:
# 
# - San Francisco - 2397.40 miles (the second line probably)
# - Los Angeles - 2562.87 miles (could be the first line)
# - New York - 4965.20 miles (could be the third)
# 
# So region 174 must be California and 348 New York with 48862 being New York city.
# 
# Let's look at trips from New York city.
fromny = (c66in[(c66in.user_location_region == 348) & 
                (c66in.user_location_city == 48862)]
          .groupby(['hotel_market'])
          ['orig_destination_distance']
          .agg(['min','mean','max','count'])
          .sort_values(by='count',ascending=False))
fromny.head(10)
# We can see that New York city itself is probably hotel_market 675.
# 
# Distances from New York:
# 
# - to Miami - 1093.57 miles -> hotel_market 701
# - to Las Vegas - 2230.03 miles -> hotel_market 628
# - to Los Angeles - 2448.30 miles -> hotel_market 365
# - to San Francisco - 2568.57 miles -> hotel_market 1230
# - to Chicago - 711.83 miles -> Chicago is hotel_market 637
# - to Washington - 203.78 miles -> Washington might be hotel_market 191 (?)
# - to Philadelphia - 80.63 miles -> hotel_market 623 (?)
# 
# We already know Los Angeles as a user_location_city. Let's do a check by confirming that trips from that city id to hotel market 365 have small distances.
(c66in[(c66in.hotel_market==365) & 
       (c66in.user_location_region==174) & 
       (c66in.user_location_city==24103)]
 ['orig_destination_distance'].describe())
# This looks about right!
# 
# ## Going international
# 
# Let's check international trips to and from New York.
tony = (train[(train.hotel_market == 675) & (train.user_location_country != 66)]
        .groupby(['user_location_country','user_location_region', 'user_location_city'])
        ['orig_destination_distance']
        .agg(['min','mean','max','count'])
        .sort_values(by='count',ascending=False))
tony.head(10)
# People are flying to New York a lot from country 205. 
# 
# Most popular city is 342 miles away so that must be Toronto (distance from NY 342.42 miles). So user_location_country 205 must be Canada. It is also hotel_country 198 as seen from the very first table. We can figure out other canadian cities and regions by their distances from NY.
# 
# Next is user_location_country 1 with mean distance 4283 miles. That would probably be somewhere in Europe. Some poking around the map brings up Rome at 4286.00 miles from NY. Then there should be another italian city at 4019 miles. Milan comes up with 4020.99 miles. So maybe user_location_country 1 is Italy.
# 
# Next user_location_country is 215, looks like this is Mexico, with distance to Mexico city being 2089.76 miles. Then Mexico is also hotel_country 8.
# 
# How about the trips from New York?
fromny = (train[(train.hotel_country != 50) & 
                (train.user_location_country == 66) &
                (train.user_location_region == 348) &
                (train.user_location_city == 48862)]
        .groupby(['hotel_country','hotel_market'])
        ['orig_destination_distance']
        .agg(['min','mean','max','count'])
        .sort_values(by='count',ascending=False))
fromny.head(10)
# First row - with hotel_market 110 - is some place in Mexico - there's a city called Cancún at 1548.30 miles from NY.
# 
# Second line looks like London (3465.05 miles), so hotel_country 70 UK, hotel_market 19 London.
# 
# Lines 3-5 judging by the distance should be somewhere in the Caribbean.
# 
# The next line might be Paris (3631.16 miles), so hotel_country 204 France, hotel_market 27 Paris.
# 
# ## Results
# 
# It looks like it's completely possible to deanonymize the countries and cities in this dataset. At least the popular ones. The more countries and cities we identify the easier the subsequent task becomes. We could sort of triangulate the locations yet uncovered using distances to already known locations.
# 
# Ways to use this information:
# 
# - classify trips as home or abroad
# - classify destinations as sea-side, ski resorts, historical cities and so on.
# - what else?
# 
# Here is a list of countries and cities matched so far. 
# 
# *removed to not tempt anyone*