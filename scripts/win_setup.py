import urllib.request
import zipfile
import os
import json

def download_energyplus():
    version = '25.2.0'
    print(f"Fetching EnergyPlus {version} releases...")
    req = urllib.request.Request("https://api.github.com/repos/NREL/EnergyPlus/releases")
    with urllib.request.urlopen(req) as response:
        releases = json.loads(response.read().decode())
        
    url = None
    for release in releases:
        if release['tag_name'] == f'v{version}':
            for asset in release['assets']:
                if 'Windows-x86_64.zip' in asset['name']:
                    url = asset['browser_download_url']
                    break
    
    if not url:
        print("Could not find EnergyPlus Windows release.")
        return

    os.makedirs('vendor', exist_ok=True)
    zip_path = 'vendor/energyplus.zip'
    print(f"Downloading EnergyPlus from {url}...")
    urllib.request.urlretrieve(url, zip_path)
    print("Extracting EnergyPlus...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('vendor')
    os.remove(zip_path)
    print("EnergyPlus extracted.")

def download_weather():
    url = "https://climate.onebuilding.org/WMO_Region_2_Asia/IND_India/TN_Tamil_Nadu/IND_TN_Chennai.Intl.AP.432790_TMYx.2009-2023.zip"
    os.makedirs('models/weather', exist_ok=True)
    zip_path = 'models/weather/weather.zip'
    print(f"Downloading Weather Data from {url}...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
            out_file.write(response.read())
        print("Extracting Weather Data...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('models/weather')
        os.remove(zip_path)
        print("Weather data extracted.")
    except Exception as e:
        print(f"Error downloading weather: {e}")

if __name__ == "__main__":
    download_energyplus()
    download_weather()
