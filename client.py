import requests

# Fake news sample
data = {
    'title': 'Scientists Discover New Species of Rainbow-Colored Butterflies',
    'text': 'In a groundbreaking scientific discovery, researchers have identified a previously unknown species of butterflies that are uniquely rainbow-colored. These stunning butterflies, which have never been observed before, are believed to inhabit remote rainforests in an undisclosed location. The scientists describe them as "a mesmerizing spectacle of nature" and are excited to learn more about their behavior and habitat. The discovery has sparked interest worldwide among butterfly enthusiasts and nature lovers.',
    'subject': 'scienceNews',
    'date': 'September 30, 2023'
}

# # True news sample
# data = {
#     'title': 'Senate panel votes to advance tax bill',
#     'text': 'WASHINGTON (Reuters) - The U.S. Senate Budget Committee voted along party lines on Tuesday to send a Republican tax bill to the full Senate for a vote. The 12-to-11 vote “moves us one step closer to a simpler, fairer, and more transparent tax system,” Budget Committee Chairman Mike Enzi said in a statement.  The full Senate is expected to begin debating the tax bill and vote on it sometime this week. The Republican-controlled House of Representatives has already passed its version of a package of tax cuts. ',
#     'subject': 'politicsNews',
#     'date': 'November 28, 2017'
# }

response = requests.post('http://localhost:5000/predict', json=data)

prediction = response.json()['prediction']
print(prediction)
