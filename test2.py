from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

# 1. The address string
address = "Arizona Cancer Center at UMC North, Tucson, Arizona, 85719, United States"

# 2. Initialize the geolocator
geolocator = Nominatim(user_agent="my_geocoder_application_v1.1") # Update version or app name

try:
    # 3. Geocode the address
    location = geolocator.geocode(address, timeout=10)

    # 4. Print the results and generate Google Maps link
    if location:
        print(f"Input Address: {address}")
        print(f"Found Address: {location.address}")
        print(f"Latitude: {location.latitude}")
        print(f"Longitude: {location.longitude}")

        # Generate Google Maps link
        latitude = location.latitude
        longitude = location.longitude
        google_maps_link_simple = f"https://www.google.com/maps?q={latitude},{longitude}"
        google_maps_link_search = f"https://www.google.com/maps/search/?api=1&query={latitude},{longitude}"
        google_maps_link_zoom = f"https://www.google.com/maps/@{latitude},{longitude},17z" # 17z is a good zoom level

        print(f"\n--- Google Maps Links ---")
        print(f"Simple Link: {google_maps_link_simple}")
        print(f"Search Link (often good for markers): {google_maps_link_search}")
        print(f"Link with Zoom: {google_maps_link_zoom}")

    else:
        print(f"Location not found for: {address}")

except GeocoderTimedOut:
    print("Error: Geocoding service timed out.")
except GeocoderUnavailable:
    print("Error: Geocoding service unavailable. Check your internet connection or the service status.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    
# https://www.google.com/maps?q=32.2643805,-110.9497108