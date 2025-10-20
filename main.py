import json
from typing import Optional
from pydantic import BaseModel, Field
from genkit.ai import Genkit
from genkit.plugins.google_genai import GoogleAI
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Genkit with the Google AI plugin
ai = Genkit(
    plugins=[GoogleAI()],
    model='googleai/gemini-2.5-flash',
)
# Define input schema
class RecipeInput(BaseModel):
    ingredient: str = Field(description='Main ingredient or cuisine type')
    dietary_restrictions: Optional[str] = Field(default=None, description='Any dietary restrictions')

# Define output schema
class Recipe(BaseModel):
    title: str
    description: str
    prep_time: str
    cook_time: str
    servings: int
    ingredients: list[str]
    instructions: list[str]

# Define a recipe generator flow
@ai.flow()
async def recipe_generator_flow(input_data: RecipeInput) -> Recipe:
    # Create a prompt based on the input
    dietary_restrictions = input_data.dietary_restrictions or 'none'

    prompt = f"""Create a recipe with the following requirements:
        Main ingredient: {input_data.ingredient}
        Dietary restrictions: {dietary_restrictions}"""

    # Generate structured recipe data using the same schema
    result = await ai.generate(
        prompt=prompt,
        output_schema=Recipe,
    )

    if not result.output:
        raise ValueError('Failed to generate recipe')

    return result.output

async def main() -> None:
    # Run the flow
    recipe = await recipe_generator_flow(RecipeInput(
        ingredient='avocado',
        dietary_restrictions='vegetarian'
    ))

    # Print the structured recipe
    print(json.dumps(recipe, indent=2))

ai.run_main(main())