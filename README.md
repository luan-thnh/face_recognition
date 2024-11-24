# Face Recognition Check-in App

This is a face recognition check-in application. The app uses FastAPI for the backend and Next.js for the frontend. You can upload photos to the backend for face recognition and check-in, as well as manage machine learning models and collect data.

## Directory Structure

```
/face_recognition
├── /client                 # Frontend directory (Next.js)
│   └── ...                 # Frontend files for the Next.js application
├── /server                 # Backend directory (FastAPI)
│   ├── main.py             # Main FastAPI application file
│   ├── face_recognition.py # File containing face recognition logic
│   ├── collect_data.py     # Data collection and processing
│   └── train_model.py      # Training the face recognition model
└── README.md               # Installation and usage guide
```

## Requirements

- Python 3.7 or higher
- Node.js and npm (for the Next.js frontend)
- A virtual environment for the backend (recommended)
- Required Python and npm libraries for the application

## Installation

### 1. Install Backend (FastAPI)

1. **Navigate to the `server` directory:**

   ```bash
   cd server
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**

   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install the required libraries:**

   ```bash
   pip install -r requirements.txt
   ```

   Or, if there is no `requirements.txt` file, install the libraries directly:

   ```bash
   pip install fastapi uvicorn torch torchvision pillow numpy
   ```

5. **Run the FastAPI server:**

   From the `server` directory, run the FastAPI server using `uvicorn`:

   ```bash
   uvicorn main:app --reload
   ```

   - `main`: the name of the file containing the FastAPI app (`main.py`).
   - `--reload`: enables auto-reloading of the server when code changes (for development).

   The server will run at `http://127.0.0.1:8000`.

### 2. Install Frontend (Next.js)

1. **Navigate to the `client` directory:**

   ```bash
   cd client
   ```

2. **Install the required libraries for Next.js:**

   ```bash
   npm install
   ```

3. **Run the Next.js app:**

   ```bash
   npm run dev
   ```

   The frontend app will run at `http://localhost:3000`.

## Running the Project

After successfully installing both the backend and frontend, you can test the app by following these steps:

1. **Run the backend:**

   The backend will accept photos and perform face recognition. You can check the API documentation at `http://127.0.0.1:8000/docs`.

2. **Run the frontend:**

   The frontend will allow users to upload photos and display the check-in result. You can access the frontend at `http://localhost:3000`.

## API Endpoints

### POST `/checkin/`

- **Description:** Receives a photo from the user, performs face recognition, and returns the result.
- **Request:** A JPEG or PNG image sent as `multipart/form-data`.
- **Successful Response:**
  ```json
  {
    "status": "success",
    "person_id": "person_1",
    "confidence": 0.92
  }
  ```
- **Failed Response:**
  ```json
  {
    "status": "fail",
    "reason": "low confidence"
  }
  ```

## Face Recognition Model Management

### Collect Data

The application supports face data collection through the `collect_data.py` script. You can run this script to collect data for training the face recognition model.

```bash
python collect_data.py
```

### Train Model

Once you have collected sufficient data, you can use the `train_model.py` script to train the face recognition model.

```bash
python train_model.py
```

### Using the Trained Model

The trained face recognition model will be saved as the file `face_recognition_model.pth`. This file will be loaded and used during face recognition.

## Next Steps

- Improve the accuracy of the face recognition model by collecting more data.
- Optimize the user interface (UI) in the frontend with Next.js.
- Deploy the app to production (e.g., on Heroku, AWS, DigitalOcean).

## License

This project is licensed under the MIT License. [LICENSE](LICENSE)
