# Train Your Classifier

This project provides a simple way to train a machine learning model for image classification using Python and TensorFlow. First, you'll set up the environment and configure Jupyter Notebook to make it easy to experiment with and visualize data. After preparing your dataset, you'll train a TensorFlow model that can learn to classify images into various categories. Once the model is trained, you can integrate it into your project to call functions that classify images automatically, making it useful for pipelines and various applications where automated image recognition is beneficial. Additionally, you could enhance the model's capabilities by employing transfer learning, utilizing pre-trained models to improve accuracy and efficiency with minimal effort.

## Python Version

This project runs on **Python 3.7**. You can use [pyenv](https://github.com/pyenv/pyenv) to manage multiple Python versions and switch between them. For more details, see the [official pyenv documentation](https://github.com/pyenv/pyenv#installation).

## Configuring Jupyter Notebook

To use Jupyter Notebook within a virtual environment, follow these steps:

1. **Create a Virtual Environment**

   Use `virtualenv` to create a virtual environment with Python 3.7:

   ```bash
   virtualenv --python=/path/to/your/python3.7 <your_env_name>

Replace /path/to/your/python3.7 with the path to your Python 3.7 executable, and <your_env_name> with a name for your environment.

2. Activate the Virtual Environment

Activate your virtual environment using:

```bash
source your_env_name/bin/activate
```

3. Install Jupyter and Required Packages

Install Jupyter Notebook and any other necessary packages:

```bash
pip install jupyter
```

4. Add a New Kernel to Jupyter (Optional)

If you want to use the virtual environment as a kernel in Jupyter Notebook, run:


```bash
ipython kernel install --user --name=<your_env_name>
```

5. Install the rest of dependencies:

```bash
pip install -r requirements.txt
```

6. Start Jupyter Notebook

Launch Jupyter Notebook:

```bash
jupyter-notebook
```


## Training Your Own Model

Inside the `train` folder, there are sample images of various animal species (e.g., cats, dogs, birds). You can train the model to recognize other categories like objects or custom images by following these steps:

1. **Gather Images**

   Place images inside the `train` folder. Create subfolders for each category you want to classify. For example:

```bash
train/
├── cats/
├── dogs/
├── birds/
└── new_category/
```

Ensure each subfolder contains a representative sample of the category you want to classify.


2. **Use the Training Notebook**

Open the **train_your_model.ipynb** notebook located in the **src** folder. Follow the steps outlined in the notebook to configure and train your model:

- You can change the base model architecture to experiment with different neural networks.
- Train the model with your custom images.
- Add additional neural network layers if desired.
- Adjust the number of training epochs to find the best results.


3. **Balance Your Dataset**

Ensure you have a similar number of images in each subfolder. An unbalanced dataset might bias the model towards classes with more images, affecting classification accuracy.

4. **Training Tips**

- More images and epochs will improve accuracy.
- Balance the dataset to avoid bias.
- Experiment with different model architectures and parameters for better results.

## Saving the Model

After training the model, save it using the following command in your Jupyter Notebook:

```python
model.save('./your_models_trained/<your_model_name>.h5')
```

## Loading the Model

You can load a saved model using TensorFlow like this:

```python
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('./your_models_trained/<your_model_name>.h5')
```
