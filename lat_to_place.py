from geopy.geocoders import Nominatim

def get_district_name(latitude, longitude):
    # Initialize Nominatim API
    geolocator = Nominatim(user_agent="geoapi")
    
    # Use reverse method to get location details
    location = geolocator.reverse((latitude, longitude), exactly_one=True)
    
    # Extract district from location details
    if location and 'address' in location.raw:
        address = location.raw['address']
        district = address['state_district']
        return district
    return None
