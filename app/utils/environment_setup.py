import os
from dotenv import load_dotenv

class EnvironmentSetup:
    def __init__(self, env_path=None):
        # print pwd     
        if env_path is None:
            env_path = os.path.join(os.path.dirname(__file__), '../../.env')
            print(env_path)
        load_dotenv(env_path)
        self.REDIS_URL = os.getenv('REDIS_URL')
        self.CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL')
        self.GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
        self.LANGSMITH_TRACING = os.getenv('LANGSMITH_TRACING')  # Fixed: removed V2
        self.LANGSMITH_ENDPOINT = os.getenv('LANGSMITH_ENDPOINT')
        self.LANGSMITH_API_KEY = os.getenv('LANGSMITH_API_KEY')
        self.LANGSMITH_PROJECT = os.getenv('LANGSMITH_PROJECT')
        self.SECRET_KEY = os.getenv('SECRET_KEY')
        self.ALGORITHM = os.getenv('ALGORITHM')
        self.ACCESS_TOKEN_EXPIRE_MINUTES = os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES')

        # Set environment variables for LangSmith
        os.environ["LANGCHAIN_TRACING_V2"] = self.LANGSMITH_TRACING or "false"
        os.environ["LANGCHAIN_ENDPOINT"] = self.LANGSMITH_ENDPOINT or ""
        os.environ["LANGCHAIN_API_KEY"] = self.LANGSMITH_API_KEY or ""
        os.environ["LANGCHAIN_PROJECT"] = self.LANGSMITH_PROJECT or ""

        print("Environment variables loaded successfully.")
        print(self.as_dict())
    def as_dict(self):
        return {
            'REDIS_URL': self.REDIS_URL,
            'CELERY_BROKER_URL': self.CELERY_BROKER_URL,
            'GOOGLE_API_KEY': self.GOOGLE_API_KEY,
            'LANGSMITH_TRACING': self.LANGSMITH_TRACING,
            'LANGSMITH_ENDPOINT': self.LANGSMITH_ENDPOINT,
            'LANGSMITH_API_KEY': self.LANGSMITH_API_KEY,
            'LANGSMITH_PROJECT': self.LANGSMITH_PROJECT,
            'SECRET_KEY': self.SECRET_KEY,
            'ALGORITHM': self.ALGORITHM,
            'ACCESS_TOKEN_EXPIRE_MINUTES': self.ACCESS_TOKEN_EXPIRE_MINUTES,
        }
