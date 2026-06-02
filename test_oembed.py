import urllib.parse
import urllib.request
import json
url = "https://www.youtube.com/watch?v=Q0BOH_s9gSU"
oembed_url = f"https://www.youtube.com/oembed?url={urllib.parse.quote(url)}&format=json"
print(oembed_url)
req = urllib.request.Request(oembed_url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, timeout=5) as response:
        data = json.loads(response.read())
        print(data)
except Exception as e:
    print(type(e), e)
