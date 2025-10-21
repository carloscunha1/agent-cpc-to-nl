from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from genkit.ai import Genkit
from genkit.plugins.google_genai import GoogleAI
from dotenv import load_dotenv

load_dotenv()

# Initialize Genkit with the Google AI plugin
ai = Genkit(
    plugins=[GoogleAI()],
    model='googleai/gemini-2.5-flash',
)

app = FastAPI(title="Conversor de sentenças")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class nl_to_logic_input(BaseModel):
    natural_language: str = Field(description='Sentença em linguagem natural')
    
class logic_to_nl_input(BaseModel):
    logical_expression: str = Field(description='Fórmula do cálculo proposicional clássico')

class logical_conversion(BaseModel):
    original: str
    converted: str
    explanation: str

# Define natural language to logic flow
@ai.flow()
async def nl_to_logic_flow(input_data: nl_to_logic_input) -> logical_conversion:
    prompt = f"""Converta a seguinte sentença em linguagem natural para fórmula do cálculo proposicional clássico:
    
    Sentença: "{input_data.natural_language}"
    
    Utilize os símbolos: ∧ (e), ∨ (ou), ¬ (negação), → (implica), ↔ (se e se somente se)
    Atribua letras às proposições (p, q, r, etc)"""
    
    result = await ai.generate(prompt=prompt, output_schema=logical_conversion)
    
    if not result.output:
        raise ValueError('Failed to convert natural language to logic')
    
    return result.output

# Define logic to natural language flow
@ai.flow()
async def logic_to_nl_flow(input_data: logic_to_nl_input) -> logical_conversion:
    prompt = f"""Converta a seguinte fórmula do cálculo proposicional cássico para uma sentença em linguagem natural:
    
    Fórmula: {input_data.logical_expression}
    
    Deixe ela clara e compreensível."""
    
    result = await ai.generate(prompt=prompt, output_schema=logical_conversion)
    
    if not result.output:
        raise ValueError('Failed to convert logic to natural language')
    
    return result.output

# API Endpoints
@app.post("/convert/nl-to-logic", response_model=logical_conversion)
async def convert_nl_to_logic(input_data: nl_to_logic_input):
    return await nl_to_logic_flow(input_data)

@app.post("/convert/logic-to-nl", response_model=logical_conversion)
async def convert_logic_to_nl(input_data: logic_to_nl_input):
    return await logic_to_nl_flow(input_data)

@app.get("/")
async def root():
    return {"message": "Logic Converter API - Use POST /convert/nl-to-logic or /convert/logic-to-nl"}