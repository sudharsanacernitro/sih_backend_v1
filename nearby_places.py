import requests

def get_nearby_villages(lat, lon, radius, api_key):
    base_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        'location': f'{lat},{lon}',
        'radius': radius,  # Radius in meters (15000 meters = 15 km)
        'type': 'locality',  # You can use "locality" to get towns/villages
        'key': api_key
    }
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        results = response.json()['results']
        if results:
            for place in results:
                print(f"Place: {place['name']}, Location: {place['geometry']['location']}")
        else:
            print("No villages or towns found within the specified radius.")
    else:
        print(f"Error in the request: {response.status_code}")

# Replace 'YOUR_API_KEY' with your actual Google API key
api_key = 'AIzaSyBzasKUAdc4_8OZJGCWrntd9d3AKn5P4qE'
latitude = 11.261064266666667  # Example latitude (Delhi, India)
longitude = 77.66910678333333  # Example longitude (Delhi, India)
radius = 150000  # 15 km in meters
get_nearby_villages(latitude, longitude, radius, api_key)
