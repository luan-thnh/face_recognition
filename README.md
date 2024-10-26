# Face Recognition Project

This project implements a face recognition system using convolutional neural networks (CNNs) with TensorFlow and OpenCV. It includes functionality for collecting face data, training a model, and visualizing performance metrics.

## Table of Contents

- [Face Recognition Project](#face-recognition-project)
  - [Table of Contents](#table-of-contents)
  - [Requirements](#requirements)
    - [Required Libraries](#required-libraries)
  - [Setup Instructions](#setup-instructions)
  - [How to Run](#how-to-run)
    - [Collect Face Data](#collect-face-data)
    - [Train the Model](#train-the-model)
  - [Directory Structure](#directory-structure)
  - [License](#license)

## Requirements

- Python 3.x
- pip

### Required Libraries

The required libraries for this project are listed in `requirements.txt`. You can install them using the following command after creating a virtual environment.

## Setup Instructions

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/luan-thnh/face_recognition.git
   cd face_recognition
   ```

2. **Create a Virtual Environment**:

   To create a virtual environment, run the following command:

   ```bash
   python -m venv .venv
   ```

3. **Activate the Virtual Environment**:

   - On Windows:

     ```bash
     .venv\Scripts\activate
     ```

   - On macOS/Linux:

     ```bash
     source .venv/bin/activate
     ```

4. **Install Required Libraries**:

   After activating the virtual environment, install the required libraries using:

   ```bash
   pip install -r requirements.txt
   ```

## How to Run

### Collect Face Data

To collect face data, run the following command:

```bash
python collect_data.py
```

You will be prompted to enter a Face ID, and the program will start capturing images from your camera. Press the ESC key to stop the process.

### Train the Model

Once you have collected the data, you can train the model by running:

```bash
python train_model.py
```

This will load the collected data, train the face recognition model, and save the trained model in the `./model/` directory. Training metrics will be visualized and saved in the `images` directory.

## Directory Structure

```
.
├── collect_data.py       # Script to collect face data
├── train_model.py        # Script to train the face recognition model
├── requirements.txt      # List of required libraries
├── datasets              # Directory to store face images
├── model                 # Directory to save the trained model
└── images                # Directory to save training visualizations
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
