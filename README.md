# Flower Image Classifier

In this project we built image classifier model to predict flower image. Then we provide command line to predict class of flower on image provided.

### Data

The data for this project is quite large - in fact, it is so large you cannot upload it onto Github.  If you would like the data for this project, you will want download it from the workspace in the classroom.  Though actually completing the project is likely not possible on your local unless you have a GPU.  You will be training using 102 different types of flowers, where there ~20 images per flower to train on.  Then you will use your trained classifier to see if you can predict the type for new images of the flowers.


### How to run

To run this application please follow these steps below:

- Install python 3.7.
- Install all dependencies using `pip install -r requirements.txt`
- Run `python path/to/image/file path/to/keras/model --top_k k --category_names label_map.json`.
- I provide sample for keras model on this project. If you want to use this model you can provide model1589354751.h5 on previous step. But you are free to use your own model.