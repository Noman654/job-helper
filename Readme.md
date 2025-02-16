### README.md

```markdown
# Advanced Resume Ranking System with Gemini Integration

This project is an API for extracting criteria from job descriptions and scoring resumes using Gemini integration. It leverages FastAPI for building the web service and provides endpoints for extracting criteria and scoring resumes.

## Features

- Extract ranking criteria from job descriptions (PDF or DOCX).
- Score multiple resumes against provided criteria.
- Supports different modes for LLM usage and scoring methods.
- Returns results in an Excel (CSV) format.

## Requirements

- Python 3.7 or higher
- FastAPI
- Uvicorn
- Pandas
- Other dependencies as specified in `requirements.txt`

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/ass-ai-engineer.git
   cd ass-ai-engineer/job-helper
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

To run the FastAPI application, use the following command:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

You can access the API documentation at `http://localhost:8000/docs`.

## Docker

To run the application in a Docker container, follow these steps:

1. Build the Docker image:

   ```bash
   docker build -t resume-ranking-system .
   ```

2. Run the Docker container:

   ```bash
   docker run -d -p 8000:8000 resume-ranking-system
   ```

You can access the API documentation at `http://localhost:8000/docs`.

## API Endpoints

### Extract Criteria

- **POST** `/extract-criteria`
- Accepts a job description file and extracts key ranking criteria.

### Score Resumes

- **POST** `/score-resumes`
- Scores multiple resumes based on provided criteria and returns an Excel sheet with results.

### Health Check

- **GET** `/health`
- Verifies the API is running and returns configuration information.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

### Dockerfile

```dockerfile
# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Notes:
- Make sure to create a `requirements.txt` file that lists all the necessary dependencies for your FastAPI application.
- Replace `yourusername` in the README with your actual GitHub username or organization name.
- Adjust the Python version in the Dockerfile if necessary, based on your application's requirements.