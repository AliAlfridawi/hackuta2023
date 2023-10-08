#This is code showing the basics of how to interface with the model
#Should be implemented with bryan's python flask stuff

from roboflow import Roboflow
rf = Roboflow(api_key="zHtICeX8iyzeTtCNzHgI")
project = rf.workspace().project("water-damage-finder")
model = project.version(2).model

photoName = "temp.jpg"
# infer on a local image
print(model.predict(photoName, confidence=40, overlap=30).json())

# visualize your prediction
model.predict(photoName, confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())