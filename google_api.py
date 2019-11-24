# Python program to get a set of  
# places according to your search  
# query using Google Places API 

# importing required modules 
import requests
import json

# enter your api key here 
api_key = ''

# url variable store url 
url = "https://maps.googleapis.com/maps/api/place/textsearch/json?"


# get method of requests module 
# return response object 

def get(query): 

    r = requests.get(url + 'query=' + query +
                    '&key=' + api_key)

    # json method of response object convert 
    #  json format data into python format data 
    x = r.json()

    # now x contains list of nested dictionaries 
    # we know dictionary contain key value pair 
    # store the value of result key in variable y 
    y = x['results']


    place = {}
    places = []


    # keep looping upto lenght of y 
    for i in range(len(y)):
        # Print value corresponding to the
        # 'name' key at the ith index of y

        try:
            if y[i]["price_level"] == 1:
                price = "$"
            elif y[i]["price_level"] == 2:
                price = "$$"
            elif y[i]["price_level"] == 3:
                price = "$$$"
            elif y[i]["price_level"] == 4:
                price = "$$$$"
            else:
                price = "N/A"
        except:
            price = "N/A"

        place = {
            "name": y[i]["name"],
            "address": y[i]["formatted_address"],
            "price": price
        }

        places.append(place)
    return places;
