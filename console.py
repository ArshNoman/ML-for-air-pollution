import pandas


df = pandas.read_csv('Bishkekdata.csv')


def get_level_of_concern(aqi):
    if aqi <= 50:
        return '0'
    elif aqi <= 100:
        return '1'
    elif aqi <= 150:
        return '2'
    elif aqi <= 200:
        return '3'
    elif aqi <= 300:
        return '4'
    else:
        return '5'

# ,,
df['variable wind direction'] = False


# df = df.drop('north-northwest', axis=1)
df.to_csv('Bishkekdata.csv', index=False)
