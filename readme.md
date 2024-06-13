# Emotion Detection Using Hugging Face and Gradio

## Project Overview

- This project aims to develop an application that detects and classifies emotions in text inputs.
- I use a pre-trained model on the Emotion dataset, to showcase the capabilities of Hugging Face models and Gradio interfaces.
- The entire workflow is consolidated within a single Jupyter Notebook, so that anyone can adapt and experiment with it.
- I have added detailed comments and helpful tips to the code so that you can parse it without any specialized AI/ML knowledge.
- This project reflects my journey and learning in the AI/ML domain, showcasing my ability to apply these technologies to real-world scenarios.

## Motivation

- I enjoy evaluating different AI/ML models and checking out their capabilities.
- The [Emotion dataset on Hugging Face](https://huggingface.co/datasets/dair-ai/emotion) contains English Twitter messages labeled with six basic emotions: `anger, fear, joy, love, sadness, and surprise`.
- I found it interesting and used a pre-trained model, fine-tuned it on this dataset, and then used [Gradio](https://www.gradio.app) to build a user interface and deploy it on [Hugging Face](https://huggingface.co).
- The [fine-tuned model](https://huggingface.co/kanad13/emotion_detection/tree/main) may not be the best due to resource constraints during training. However, you can achieve better performance by executing the [Jupyter Notebook](/Emotion_Detection_App.ipynb) with more better resources.
- My motivation was to create a modular Jupyter Notebook that can be quickly modified by me or others to check out other models too.
- [Here is the link](https://huggingface.co/spaces/kanad13/emotion-detection_app) to the Emotion Detection App if you would like to try it first hand.

## Ease of use

- While there may be similar projects already developed, my goal is to consolidate the entire application within a single Jupyter Notebook.
- This makes it easier for others and myself to adapt the code for using different models, changing hyperparameters, and fine-tuning it further.
- I have put detailed comments throughout the code to ensure that it is understandable and adaptable.
- Furthermore, it can work out-of-the-box on Google Colab or Apple Silicon Macs.

## Tools & Technologies used

- **Hugging Face Transformers**: A library providing thousands of pre-trained models for Natural Language Processing (NLP) tasks. It simplifies the process of using these models for various applications.
- **Hugging Face Spaces**: A platform to host and share machine learning applications. It provides an easy way to deploy and share ML models with the community.
- **Gradio**: An open-source Python library that allows you to quickly create customizable user interfaces for machine learning models. It simplifies the deployment and sharing of ML models.
- **PyTorch**: An open-source machine learning library for Python, used for applications such as computer vision and natural language processing. It provides a flexible and efficient platform for deep learning.
- **Pandas**: A powerful data manipulation and analysis library for Python. It is essential for data preprocessing and exploratory data analysis.
- **Matplotlib**: A plotting library for creating static, animated, and interactive visualizations in Python. It is used for visualizing data distributions and model performance.

## Usage Instructions

- The code in this repository can be executed in 3 ways:
  - on google colab
  - using Python virtual environment
  - using VSCode DevContainers
- See below detailed steps for each option.

### Google Colab

- It is easiest to run this app in Google Colab. For detailed instructions on how to do it, check the [Section 1 of the Jupyter Notebook](Emotion_Detection_App.ipynb).

### Python Virtual Environment

- All packages needed for running this code can be deployed inside a virtual environment
- These are the steps
  - First clone this repo locally
  - Then open the folder
    - `cd emotion_detection`
- Then create a Python virtual environment
  - `python -m venv emotion_detection_venv`
- Then activate the virtual environment
  - On windows - `.\emotion_detection_venv\Scripts\activate`
  - On mac - `source emotion_detection_venv/bin/activate`
  - And now install all dependencies
    - `pip install -r .devcontainer/requirements.txt`

### VSCode DevContainers

- If you use [vscode devcontainers](https://code.visualstudio.com/docs/devcontainers/containers) like me, then you dont have to do anything noted above in the python virtual env section
- I have setup the repo nicely to be launch-ready the moment you download it
- Launch vscode, and then open the command palette (ctrl+shift+p), and then select "Remote-Containers: Open Folder in Container"
- Navigate to the cloned repository folder and bam...you are done...all requirements will be automatically installed
- You can also make modifications to these 2 files as needed
  - [devcontainer.json](.devcontainer/devcontainer.json)
  - [postStart.sh](.devcontainer/postStart.sh)

### Optional - Run Gradio locally

- I have developed the UI for the app using Gradio. To use it, execute the following command:

```bash
python app.py
```

- The app will be accessible at `http://localhost:7860`.

## Project Workflow

### Step 1: Dataset Selection and Exploration

- **Dataset**: Emotion dataset from Hugging Face, which contains English Twitter messages labeled with six emotions: anger, fear, joy, love, sadness, and surprise.
- **Exploration**: In this section I explore the dataset to understand its structure, including the different classes of emotions and their distribution.

### Step 2: Model Selection

- **Model**: I selected a pre-trained transformer model (DistilBERT) that is suitable for text classification tasks.
- **Justification**: I chose DistilBERT for its efficiency and performance. It retains 97% of BERT's capabilities while being smaller and faster. I have also added a comparison of different models inside the Jupyter Notebooks.

### Step 3: Model Fine-tuning

- **Objective**: In this section I fine-tuned the pre-trained DistilBERT model on the Emotion dataset to enhance its performance for emotion detection.
- **Steps**:
  - Load and preprocess the Emotion dataset.
  - Fine-tune the DistilBERT model on the training dataset.
  - Save the fine-tuned model and tokenizer.

### Step 4: Model Evaluation

- **Objective**: In this section, I evaluate the fine-tuned model's performance on a test set.
- **Metrics**: I calculate and display accuracy, precision, recall, and F1-score for each emotion class.

### Step 5: Upload Model to Hugging Face

- **Objective**: In this section, I uploaded the fine-tuned model to Hugging Face for easy access and sharing.
- **Steps**:
  - Log in to Hugging Face account.
  - Create a new repository on Hugging Face.
  - Push the fine-tuned model and tokenizer to the repository.

### Step 6: Gradio Interface Development

- **Objective**: In this section, I coded a user-friendly Gradio interface for the emotion detection model.
- **Steps**:
  - Create a Gradio interface that accepts text input and displays the predicted emotion.
  - Launch the Gradio app locally for testing.

### Step 7: Deployment on Hugging Face Spaces

- **Objective**: In this section, I deployed the Gradio application on Hugging Face Spaces so that others can access it too.
- **Steps**:
  - Create a new Space on Hugging Face.
  - Clone the Space repository.
  - Add the Gradio app script and requirements.
  - Push the changes to the Space repository.

## Conclusion

This project provides a comprehensive guide to building an emotion detection application using state-of-the-art NLP models and deploying it with a user-friendly interface. By consolidating the entire workflow in a single Jupyter Notebook, I aim to make the process accessible and modifiable for anyone interested in exploring emotion detection or similar NLP tasks.

## Acknowledgements

- **Hugging Face**: For providing the pre-trained models and datasets.
- **Gradio**: For the user-friendly interface development tools.
- **Google Colab**: For providing free GPU resources for model training.
