# paper-chatbot-rag  
A simple application that allows users to upload a paper and chat with the paper.  

## Setup Instructions  

1. Add an `.env` file to the `backend` directory and include the following line:  
 `OPENAI_API_KEY="YOUR API KEY HERE"`

2. Clone the repository:  
```bash  
git clone [REPO_URL]  
```  
3. Run the application using Docker Compose:
```bash
docker-compose up -d
cd paper-chatbot-rag
```

Access the frontend at http://127.0.0.1:7860.

Enjoy chatting with your papers!
