import os
import json
import mimetypes
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi import APIRouter, File, UploadFile, HTTPException

load_dotenv()


router1 = APIRouter()


API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")

genai.configure(api_key=API_KEY)

generation_config = {
    "response_mime_type": "application/json",
    "temperature": 0,
    "top_p": 1,
    "top_k": 1,
    "candidate_count": 1
}

model = genai.GenerativeModel(
    "gemini-2.5-flash-preview-05-20", 
    generation_config=generation_config,
)



@router1.get("/health")
async def health():
    """Provides a simple health check."""
    return {"status": "ok", "message": "MODT and NOI index Backend is running"}

@router1.get('/')
async def status():
    """Provides a simple status check to confirm the backend is running."""
    return {"message": "The Docuguard Backend is running"}

@router1.post('/api/verify/noi_index')
async def verify_noi_documents(
    originalFile: UploadFile = File(...),
    suspectedFile: UploadFile = File(...)
):
    """
    API endpoint to verify NOI INDEX II documents using FastAPI's UploadFile.
    """
  
    if originalFile.filename == '' or suspectedFile.filename == '':
        raise HTTPException(status_code=400, detail="One or both files are empty")

    
    original_data = await originalFile.read()
    suspected_data = await suspectedFile.read()

   
    original_mime_type = mimetypes.guess_type(originalFile.filename)[0] or 'application/octet-stream'
    suspected_mime_type = mimetypes.guess_type(suspectedFile.filename)[0] or 'application/octet-stream'

    original_blob = {'mime_type': original_mime_type, 'data': original_data}
    suspected_blob = {'mime_type': suspected_mime_type, 'data': suspected_data}

    prompt = """
    You are a highly skilled forensic document examiner, multilingual OCR analyst, and forgery detection expert.
    
    You will receive two government-issued documents — an "Original Document" and a "Suspected Document".
    The documents are structured similarly but may:
    - Contain different languages (English ↔ Marathi).
    - Have columns in different positions.
    - Use alternate wording for the same field.
    
    here is the list of columns you need to extract from the index2 document and do the content comparison.
    list of columns - [District,Taluka,village name ,Title ,Loan Amount ,Property Description,Area ,Mortgagor Name,Address,Mortgagee Name,Address,Date of Mortgage ,Date of Filing,Filing No,Stamp Duty ,Filing Fees ,Date of Submission . ]

    your output should be simple and brief summary of the findings.
    
    ---
    ## **Preprocessing & OCR Instructions**

1. **OCR & Translation** - Apply OCR with multilingual support (English + Marathi).  
   - Translate Marathi headers/labels into English.  
   - Normalize field names to a common schema.

2. **Column Matching** - Identify **paired columns** (same meaning, different language).  
   - Align columns even if their **order/placement is different**.  
   - Example1: "Area" ↔ "क्षेत्रफळ" ,"Stamp Duty"↔"मुद्रांक शुल्क","Date of Submission"↔"दस्तऐवज करून दिल्याचा दिनांक","Date of Mortgage"↔"दस्तऐवज करून दिल्याचा दिनांक","Loan Amount"↔"बाजारभाव (भाडेपट्ट्याच्या बाबतीतपट्टकार आकारणी देतो कि पट्टेदार ते नमूद करावे)","Property Description"↔"भूमापन ,पोटिहिस्सा व घरक्रमांक (असल्यास)".
   - Example2: "Mortgagor Name Address"↔"दस्तऐवज करून घेणाऱ्या पक्षकारांचे नाव किंवा दिवाणी न्यायालयाचा हुकूमनामा किंवा आदेश असल्यास प्रतिवादीचे नाव व पत्ता",
   - Example3: "Mortgagee Name Address"↔"दस्तऐवज करून देणाऱ्या लिहून ठेवणाऱ्या पक्षकारांचे नाव किंवा दिवाणी न्यायालयाचा हुकूमनामा किंवा आदेश असल्यास प्रतिवादीचे नाव व पत्ता".
   - Ignore columns like-["मोबदला"]

3. **Value Comparison** - Compare paired values **numerically/textually**.  
   - Allow formatting differences (e.g., "Rs.55,00,000/-" vs "5500000","dd-mm-yyyy" vs "dd/mm/yyyy").  
   - Ignore the fromating differences in amounts only consider the difference in the absoulte amount value.


---

## **Analysis Dimensions**

1. **Text & Values** - Perform **cell-by-cell comparison** after column alignment.  
   - Highlight mismatches in **names, dates, amounts, property details**.  
   - Flag altered **spellings, numbers, or translations**.  

2. **Tables & Structure** - Detect missing/extra rows or misaligned fields.  
   - Ensure data consistency across both languages.  

3. **Layout & Metadata** - Check alignment of tables, headers, margins.  
   - Detect file metadata differences (creation date, DPI, etc.).  

4. **Stamps** - Compare **seals, orientation, placement, and clarity**.  
   - Detect copy-paste or digital manipulation.  

5. **Handwritten Marks** - Compare **pattern similarity only** (SSIM > 85% = authentic).  
   - Do not interpret the text.  

---

## **Verdict Rules**

- **Authentic** → All values match (allowing for language/format differences).  
- **Forged** → Any material mismatch in values (names, amounts, property, dates, signatures, stamps).  
- **Inconclusive** → Poor scan quality or unreadable sections. 

---
    ---
    
    ### **Output Format (JSON)**
    ```json
    {
      "verdict": "Authentic" | "Forged" | "Inconclusive",
      "summary": {
        "critical_findings": "<integer>",
        "moderate_findings": "<integer>",
        "total_discrepancies": "<integer>"
      },
      "analysis_details": [
        {
          "area": "Signature" | "Text Content" | "Stamp" | "Layout" | "Image" | "Handwritten Notes/Annotations",
          "discrepancies": [
            {
              "description": "Detailed description of the specific issue found.",
              "severity": "Low" | "Medium" | "High"
            }
          ]
        }
      ]
    }
    ```
    """

    try:
        response = model.generate_content([prompt, "Original Document:", original_blob, "Suspected Document:", suspected_blob])
        report_data = json.loads(response.text)
       
        return report_data
    except Exception as e:
        print(f"An error occurred: {e}")
      
        raise HTTPException(status_code=500, detail=f"Failed to analyze documents with the AI model. Details: {str(e)}")

@router1.post('/api/verify/MODT')
async def verify_modt_documents(
    originalFile: UploadFile = File(...),
    suspectedFile: UploadFile = File(...)
):
    """
    API endpoint to verify MODT documents using FastAPI's UploadFile.
    """
    if originalFile.filename == '' or suspectedFile.filename == '':
        raise HTTPException(status_code=400, detail="One or both files are empty")

    original_data = await originalFile.read()
    suspected_data = await suspectedFile.read()

    original_mime_type = mimetypes.guess_type(originalFile.filename)[0] or 'application/octet-stream'
    suspected_mime_type = mimetypes.guess_type(suspectedFile.filename)[0] or 'application/octet-stream'

    original_blob = {'mime_type': original_mime_type, 'data': original_data}
    suspected_blob = {'mime_type': suspected_mime_type, 'data': suspected_data}

    prompt = """
    You are a highly skilled forensic document examiner, multilingual OCR analyst, and forgery detection expert.
    
    You will receive two government-issued documents — an "Original Document" and a "Suspected Document".
    The documents are structured similarly but may:
    - Contain different languages (English ↔ Marathi).
    - Have columns in different positions.
    - Use alternate wording for the same field.
    
    ---
    
  ## **Preprocessing & OCR Instructions**

1. **OCR & Translation** - Apply OCR with multilingual support (English + Marathi).  
   - Translate Marathi headers/labels into English.  
   - Normalize field names to a common schema.

2. **Column Matching** - Align columns even if their **order/placement is different**.  

3. **Value Comparison** - Compare paired values **numerically/textually**.  
   - Allow formatting differences (e.g., "Rs.55,00,000/-" vs "5500000","dd-mm-yyyy" vs "dd/mm/yyyy").  
   - Ignore the fromating differences in amounts only consider the difference in the absoulte amount value.

---

## **Analysis Dimensions**

1. **Text & Values** - Perform **cell-by-cell comparison** after column alignment.  
   - Highlight mismatches in **names, dates, amounts, property details**.  
   - Flag altered **spellings, numbers, or translations**.  

2. **Tables & Structure** - Detect missing/extra rows or misaligned fields.  
   - Ensure data consistency across both languages.  

3. **Layout & Metadata** - Check alignment of tables, headers, margins.  
   - Detect file metadata differences (creation date, DPI, etc.).  

4. **Stamps** - Compare **seals, orientation, placement, and clarity**.  
   - Detect copy-paste or digital manipulation.  

5. **Handwritten Marks** - Compare **pattern similarity only** (SSIM > 85% = authentic).  
   - Do not interpret the text.  

---

## **Verdict Rules**

- **Authentic** → All values match (allowing for language/format differences).  
- **Forged** → Any material mismatch in values (names, amounts, property, dates, signatures, stamps).  
- **Inconclusive** → Poor scan quality or unreadable sections. 

---
    
    ### **Output Format (JSON)**
    ```json
    {
      "verdict": "Authentic" | "Forged" | "Inconclusive",
      "summary": {
        "critical_findings": "<integer>",
        "moderate_findings": "<integer>",
        "total_discrepancies": "<integer>"
      },
      "analysis_details": [
        {
          "area": "Signature" | "Text Content" | "Stamp" | "Layout" | "Image" | "Handwritten Notes/Annotations",
          "discrepancies": [
            {
              "description": "Detailed description of the specific issue found.",
              "severity": "Low" | "Medium" | "High"
            }
          ]
        }
      ]
    }
    ```
    """

    try:
        response = model.generate_content([prompt, "Original Document:", original_blob, "Suspected Document:", suspected_blob])
        report_data = json.loads(response.text)
        return report_data
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze documents with the AI model. Details: {str(e)}")