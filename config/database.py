from sqlmodel import SQLModel, Field, create_engine,Session
from fastapi import FastAPI
from sqlalchemy.orm import sessionmaker 
from dotenv import load_dotenv
import os
load_dotenv()


db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_user = os.getenv("DB_USERNAME")
db_pass = os.getenv("DB_PASSWORD")
db_name = os.getenv("DB_DATABASE")


#DATABASE_URL = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

DATABASE_URL = f"mysql+pymysql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}?ssl_verify_cert=false"


engine = create_engine(DATABASE_URL,echo=True)


dblocal = sessionmaker(autocommit=False, autoflush=False, bind=engine,class_= Session )


def get_db():
    db = dblocal()
    try:
        yield db
    finally:
        db.close()








