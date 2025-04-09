# GPTeacher

GPTeacher is a Streamlit-based web application designed to help users solve programming tasks through interactive guidance and step-by-step learning using OpenAI's GPT-4o model. The chatbot utilizes an instructional scaffolding approach, encouraging active engagement rather than passive learning.

## Features

- **Test Selection:** Choose from multiple programming challenges loaded from a JSON file.
- **Stepwise Tutoring:** Instead of providing direct answers, the chatbot guides users step by step.
- **Interactive Chat:** Engage in a conversation with an AI assistant that provides guidance and answers to your coding questions.
- **Conversation History:** Keeps track of user queries to maintain context-aware assistance.
- **Korean Support:** Responds in Korean to cater to native Korean speakers.

## Installation

### Prerequisites
- Python 3.7+ installed on your system.
- OpenAI API Key (needed for GPT-4o usage).
- Streamlit (for running the web application).

### Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/username/gpteacher.git
   cd gpteacher
   ```

2. **Create a Virtual Environment**

   It's recommended to use a virtual environment to manage dependencies.

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up API Key**

   Create a `.env` file in the root directory and add:

   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

   Replace `your_openai_api_key_here` with your actual OpenAI API key.

5. **Prepare Data**

   Ensure that `data.json` exists in the root directory. The structure should be:

   ```json
   [
       {
           "title": "Sample Test 1",
           "text": "Description of the first programming task."
       },
       {
           "title": "Sample Test 2",
           "text": "Description of another coding task."
       }
       // Add more tests as needed
   ]
   ```

## Usage

Run the Streamlit web app using:

```bash
streamlit run streamlit_app.py
```

This will start the application and open it in your default web browser. If it doesn't open automatically, navigate to `http://localhost:8501` in your browser.

## Project Structure

```
GPTeacher/
│
├── .devcontainer/             # VSCode devcontainer setup  
│   ├── devcontainer.json  
│
├── app.py                     # Main chatbot application
├── data.json                  # JSON file containing programming tasks
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
|
├── .env                       # Environment variables (not committed)
```

## How GPTeacher Works
1. **Retriever** – Generates a long-term dialogue plan based on the user's input.
2. **Generator** – Produces structured responses based on the learning steps.
3. **Checker** – Evaluates user responses to determine whether to move forward, provide hints, or reset the plan.

Unlike traditional chatbots, GPTeacher **engages users in an active learning process**, ensuring they fully understand coding concepts rather than just copying solutions.
