import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=ENV_PATH)

PLAN = {
    "goal": "Validate Student Enrolment App behavior using a local open-source AI agent",
    "checks": [
        "/students",
        "/students/{student_id}",
        "/students/by-id",
        "/ask"
    ]
}

DATABASE_NAME = Path(__file__).with_name("enrolment.db")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")


def validate_student(student):
    student_id, student_name, subject_code = student

    if not isinstance(student_id, int):
        return False, "student_id must be an integer"

    if not student_name:
        return False, "student_name is required"

    if not subject_code:
        return False, "subject_code is required"

    return True, "ok"


def observe_data_quality():
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    students = cursor.execute(
        "SELECT student_id, student_name, subject_code FROM students"
    ).fetchall()

    conn.close()

    if len(students) != 10:
        return False, "Expected 10 students"

    for student in students:
        ok, msg = validate_student(student)
        if not ok:
            return False, msg

    return True, "Data validation passed"


def get_local_agent_advice(observe_message):
    prompt = (
        "You are reviewing an existing Flask Student Enrolment App.\n"
        "Current database fields: student_id, student_name, subject_code.\n"
        "Current endpoints:\n"
        "- GET /students\n"
        "- GET /students/<student_id>\n"
        "- GET /students/by-id\n"
        "- POST /ask\n\n"
        f"OBSERVE result: {observe_message}\n\n"
        "Rules:\n"
        "- Do not invent new database fields.\n"
        "- Do not invent new endpoints.\n"
        "- Do not recommend functionality that already exists.\n"
        "- Recommend one small improvement to validation, error handling, "
        "response formatting, or testing.\n"
        "- Return exactly two bullet points."
    )

    try:
        client = OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama"
        )

        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise software engineering reviewer."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=220,
            temperature=0.2,
        )

        text = response.choices[0].message.content.strip()
        return text, None

    except Exception as exc:
        return None, f"Local AI agent unavailable ({exc})."


def main():
    print("PLAN:", PLAN)

    print("ACT: Check local database records")

    ok, msg = observe_data_quality()
    print("OBSERVE:", msg)

    if not ok:
        print("ADAPT: Fix database seed data and rerun validation")
    else:
        print("ADAPT: Proceed to endpoint checks")

    advice, advice_err = get_local_agent_advice(msg)
    if advice:
        print("ADAPT (Local AI suggestion):")
        print(advice)
    else:
        print("ADAPT (Local AI suggestion):", advice_err)


if __name__ == "__main__":
    main()