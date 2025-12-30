import googlemaps
from datetime import datetime

# Initialize client
gmaps = googlemaps.Client(key='YOUR_API_KEY')

# Test: Get info about Statue of Liberty
place = gmaps.places('Statue of Liberty, NYC')
print(place['results'][0]['name'])
print(place['results'][0]['geometry']['location'])