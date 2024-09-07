import pymongo

client = pymongo.MongoClient("mongodb+srv://thapasandesh38:UQ37gQaHZSl5tjDX@cluster0.mv3gl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

db = client.sample_mflix

collection = db.movies

items = collection.find().limit(5)

for item in items:
    print(item)
