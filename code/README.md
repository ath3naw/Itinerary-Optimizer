# Description
Uses set NYC museum data and geographical locations to determine best routes to NYC museums given the starting and ending geographical coordinates (latitude, longitude) and the number of intermediate locations to visit

Use main.py to run code, will automatically call to clean_dataset.py, functions.py, and optimize.py

## Scripts
clean_dataset.py:
- always cleans dataset and adds starting and ending positions to geographical locations
- converts automatically to UTM coordinates, which calculates distances in meters

functions.py:
- gives set of necessary functions that are commonly called in other pieces of code

optimize.py:
- implements a greedy directional-based algorithm to identify best locations for itinerary to visit
    - can adjust preference for more spread out locations or nearer locations to visit (by adjusting alpha, beta, gamma)
- afterwards, calls to a dynamic programming approach to determine the best path connecting the given locations from the itinerary

## Examples
Example tests in main.py:

Enter your starting address latitude: 40.7424\
Enter your starting address longitude: -74.0061\
Enter your ending address latitude: 40.7021\
Enter your ending address longitude: -73.9921\
Enter the number of locations to visit between start and end: 11

In this next case, the starting address is the exact address of one of the museums

Enter your starting address latitude: 40.833853500753314\
Enter your starting address longitude: -73.94729768541572\
Enter your ending address latitude: 40.7021\
Enter your ending address longitude: -73.9921\
Enter the number of locations to visit between start and end: 9


