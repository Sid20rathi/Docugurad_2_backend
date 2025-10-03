from fastapi import FastAPI , APIRouter , Request, HTTPException ,Depends
from schema.schema import users , user_logs , loan_master
from config.database import get_db
from sqlmodel import Session



router3= APIRouter()

@router3.post("/api/{model_name}")
def no_of_times_model_used(model_name:str,user_name:str ,user_email:str, db: Session = Depends(get_db)):
    try:
        stmt = user_logs.insert().values(
            user_name=user_name,
            user_email=user_email,
            action_performed=model_name
        )
        db.exec(stmt)
        db.commit()
        return {"message": "Action logged successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router3.get("/")
def health_checkup():
    return {"status": "ok","message": "Model Used Update is running"}






