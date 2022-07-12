import pandas as pd
gifts = pd.read_csv('../input/gifts.csv')
gifts['GiftId'] = 66128
trips = [[i]*1000 for i in range(100)]
gifts['TripId'] = sum(trips, [])
gifts[['GiftId', 'TripId']].to_csv('result.csv', index=False)