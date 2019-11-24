# module to be used in ingredientsToRecipe and dishToRecipe

import requests


def recipe(id):
    # enter your api key here
    api_key = ''

    # url variable store url
    url = "https://api.spoonacular.com/recipes/"

    # get method of requests module
    # return response object
    r = requests.get(url + str(id) + '/information?&apiKey=' + api_key)

    print(url + str(id) + '/information?&apiKey=' + api_key)
    # json method of response object convert
    #  json format data into python format data
    x = r.json()

    y = x["extendedIngredients"]
    ingredients = []

    for i in y:
        ingredients.append(i["original"])

    return ingredients


