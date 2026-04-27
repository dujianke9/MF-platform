"""LLM Call Wrapper"""

import json
import requests
from config import DEEPSEEK_API_KEY, DEEPSEEK_API_URL, DEEPSEEK_MODEL


def call_llm(prompt: str, system_prompt: str = "", temperature: float = 0.2) -> str:
    """Call DeepSeek LLM"""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 4096,
    }
    try:
        resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[LLM Error] {e}"


def extract_operators_from_formula(formula: str) -> list:
    """
    Use LLM to extract operator list from a single formula
    """
    system_prompt = "You are a mathematical formula analysis expert."
    prompt = f"""Please analyze the following LaTeX mathematical formula and extract all operators.

Operators include but are not limited to:
- Arithmetic operators: +, -, \\cdot, 	imes, \\div, /
- Power operations: ^{{n}} (e.g., ^{{2}}, ^{{4}}, etc.)
- Fractions: \\frac
- Square root: \\sqrt
- Trigonometric functions: \\sin, \\cos, 	an, etc.
- Logarithms: \\log, \\ln
- Summation: \\sum
- Integration: \\int
- Partial derivative/differential: \\partial, d
- Comparison operators: =, <, >, \\leq, \\geq
- Other mathematical operators

Formula: {formula}

Please return the operator list in JSON array format, for example: ["\\cdot", "-", "^{{4}}", "^{{4}}"]
Note:
1. Implicit multiplication should also be represented by \\cdot
2. List each operator as many times as it appears
3. Return only the JSON array, no other content
"""
    result = call_llm(prompt, system_prompt)
    # Try to parse JSON
    try:
        # Extract JSON part
        result = result.strip()
        if result.startswith("```"):
            lines = result.split("\n")
            result = "\n".join(lines[1:-1])
        operators = json.loads(result)
        if isinstance(operators, list):
            return operators
    except json.JSONDecodeError:
        pass
    return []


def extract_physical_structures(formula: str, context: str = "") -> list:
    """
    Use LLM to extract physical structure block list from formula
    """
    system_prompt = "You are a physics and mathematical formula analysis expert."
    prompt = f"""Please analyze the following mathematical/physical formula and identify the most critical physical structure blocks.

## Requirements
1. Extract only the **most core, most physically meaningful** structure blocks, do not over-segment, be cautious of over-interpreting individual variables (like 1*some_variable, which is essentially the variable itself);
2. Each structure block should include:
   - `name`: Physical structure block name (English underscore naming convention)
   - `fragment`: Corresponding original fragment in the formula (directly excerpt the sub-expression from the formula)
   - `meaning`: Physical meaning of this structure block (brief explanation in English)

## Example
For the formula T*(1 + (kappa-1)/2 * M**2), example output:
```json
[
  {{
    "name": "stagnation_temperature_relation",
    "fragment": "T*(1 + (kappa-1)/2 * M**2)",
    "meaning": "Isentropic relation between stagnation temperature and static temperature, overall structure reflects kinetic energy conversion to thermal energy"
  }},
  {{
    "name": "compressibility_correction_factor",
    "fragment": "1 + (kappa-1)/2 * M**2",
    "meaning": "Compressibility correction factor, higher Mach number leads to more significant temperature rise"
  }},
  {{
    "name": "mach_kinetic_term",
    "fragment": "(kappa-1)/2 * M**2",
    "meaning": "Mach number kinetic term, characterizes the contribution of flow kinetic energy to temperature"
  }}
]

Formula: {formula}
{f"Context information: {context}" if context else ""}

Return only the JSON array, do not include any other content.
"""
    result = call_llm(prompt, system_prompt)
    try:
        result = result.strip()
        if result.startswith("```"):
            lines = result.split("\n")
            result = "\n".join(lines[1:-1])
        structures = json.loads(result)
        if isinstance(structures, list):
            return structures   # Return object list
    except json.JSONDecodeError:
        pass
    return []
