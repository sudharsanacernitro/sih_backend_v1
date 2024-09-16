from geopy.geocoders import Nominatim

def get_lat_long_geopy(pin_code):
    geolocator = Nominatim(user_agent="geoapiExercises")
    location = geolocator.geocode(pin_code)
    
    if location:
        return location.latitude, location.longitude
    else:
        print(f"No results found for the pin code: {pin_code}")
        return None

pin_code = '638112'  # Example pin code (for San Francisco, CA)
lat_long = get_lat_long_geopy(pin_code)

if lat_long:
    print(f"Latitude: {lat_long[0]}, Longitude: {lat_long[1]}")
else:
    print("Could not retrieve latitude and longitude.")
