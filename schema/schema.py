from sqlalchemy import MetaData,Table
from config.database import engine

from fastapi import FastAPI



metadata = MetaData()
users = Table("users", metadata, autoload_with=engine)
loan_master = Table("loan_master", metadata, autoload_with=engine)
user_logs = Table("user_logs", metadata, autoload_with=engine)







