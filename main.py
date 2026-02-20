import requests

#def call_llama(prompt):
#    response = requests.post("http://localhost:11434/api/generate",
#                             json={
#                                "model": "llama3",
#                                "prompt": prompt,
#                                "stream": True,
#                                "temprature":0
#                            })
#
#    return response


def fetch_jobs(keyword):
    url =  "https://jsearch.p.rapidapi.com/search"
    headers = {"x-rapidapi-key": "7facd78fdfmsh4b407824e737441p188d1ejsn4406fecf56f9"}

    all_jobs = []
    # for page in range(1, 11):
    #     response = requests.get( url, headers=headers, params={"query": keyword, "page": 1}).json()
    #     all_jobs += response["data"]

    print(all_jobs)

    #return all_jobs
    return requests.get( url, headers=headers, params={"query": keyword, "page": 1}).json()
