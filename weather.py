import requests

async def location_api():
    res = requests.get("http://ip-api.com/json/")
    return res.json()['lat'], res.json()['lon']

async def weather_api(coords: list):
    latitude, longitude = coords
    res = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current_weather": "true"
        }
    )
    weather = res.json()["current_weather"]
    weather_report = f"""The current weather is:
    temperature: {weather["temperature"]}
    windspeed: {weather["windspeed"]}
    wind direction: {weather["winddirection"]} degrees
    And it is {"daytime" if weather["is_day"] else "nightime"}"""
    return weather_report