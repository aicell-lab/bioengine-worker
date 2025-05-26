class DemoModel(object):
    def __init__(self):
        pass

    async def ping(self) -> str:
        return "pong"

    async def get_time(self, city_name: str) -> dict:
        from geopy.geocoders import Nominatim
        from timezonefinder import TimezoneFinder
        from datetime import datetime
        import pytz

        # Get coordinates of the city
        geolocator = Nominatim(user_agent="city_time_lookup")
        location = geolocator.geocode(city_name)

        if location is None:
            return f"City '{city_name}' not found."

        # Get timezone from coordinates
        tf = TimezoneFinder()
        timezone_str = tf.timezone_at(lng=location.longitude, lat=location.latitude)

        if timezone_str is None:
            return f"Could not determine timezone for '{city_name}'."

        # Get current time in the timezone
        timezone = pytz.timezone(timezone_str)
        city_time = datetime.now(timezone)

        return city_time.strftime(
            f"Current time in {city_name} (%Z): %Y-%m-%d %H:%M:%S"
        )
