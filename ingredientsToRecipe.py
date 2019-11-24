import requests
import idInfo
import json

# enter your api key here
api_key = ''

# url variable store url
url = "https://api.spoonacular.com/recipes/findByIngredients"

# The text string on which to search in the format x,+y,+z
def get(ingredients):
    # get method of requests module
    # return response object
    r = requests.get(url + '?ingredients=' + ingredients +
                    '&apiKey=' + api_key)

    print(url + '?ingredients=' + ingredients + '&apiKey=' + api_key)
    # json method of response object convert
    #  json format data into python format data
    x = r.json()

    food = []
    foods = []
    index = 0
    for i in x:
        if index == 2: 
            break
        try:
            food.append(i["id"])
            food.append(i["title"])
            food.append(i["image"])
            food.append(idInfo.recipe(int(i["id"])))
        except:
            print("")

        foods.append(food)
        food = []
        index += 1
    return foods

