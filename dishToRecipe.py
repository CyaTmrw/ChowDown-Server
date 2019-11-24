import requests
import idInfo
import json



# The text string on which to search in the format x%20y%20z
def get(query):
    # enter your api key here
    api_key = ''

    # url variable store url
    url = "https://api.spoonacular.com/recipes/"
        
    # get method of requests module
    # return response object
    r = requests.get(url + 'search?query=' + query + '&apiKey=' + api_key)

    # print(url + 'search?query=' + query + '&apiKey=' + api_key)
    # json method of response object convert
    #  json format data into python format data
    x = r.json()

    y = x["results"]

    food = []
    foods = []
    index = 0
    for i in y:
        if index == 2: 
            break
        try:
            food.append(i["id"])
            food.append(i["title"])
            food.append("https://spoonacular.com/recipeImages/" + i["image"])
            food.append(idInfo.recipe(int(i["id"])))
        except:
            print("")

        foods.append(food)
        food = []
        index += 1
    return foods


