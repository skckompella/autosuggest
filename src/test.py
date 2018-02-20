
import requests
from time import sleep, time

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


chars = ["w h a t".split(), "i s".split(), "y o u".split()]
req = 'http://127.0.0.1:4000/autosuggest?q='
times = []

for w in chars:
    for c in w:
        req += c
        start_time = time()
        ret = requests.get(req)
        times.append(time() - start_time)
        print(req, ret.content)
    req += "+"

print("Request times: ", times)
print("Mean request time: ", mean(times))