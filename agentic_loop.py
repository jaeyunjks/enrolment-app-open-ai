import os
import sqlite3
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI

ENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=ENV_PATH)

PLAN = {
    "goal": "Validate Student Enrolment App behavior using a local multi-agent workflow",
    "checks": [
        "/students",
        "/students/{student_id}",
        "/students/by-id",
        "/students/by-subject",
        "/ask"
    ]
}

DATABASE_NAME = Path(__file__).with_name("enrolment.db")

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434/v1"
)

IMPLEMENTATION_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5:0.5b"
)

REVIEW_MODEL = os.getenv(
    "OLLAMA_REVIEW_MODEL",
    "llama3.1:8b"
)


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
        """
        SELECT
            student_id,
            student_name,
            subject_code
        FROM students
        """
    ).fetchall()

    conn.close()

    if len(students) != 10:
        return False, "Expected 10 students"

    for student in students:
        ok, msg = validate_student(student)

        if not ok:
            return False, msg

    return True, "Data validation passed"


def observe_subject_search(subject_code):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    students = cursor.execute(
        """
        SELECT
            student_id,
            student_name,
            subject_code
        FROM students
        WHERE subject_code = ?
        """,
        (subject_code,)
    ).fetchall()

    conn.close()

    if not students:
        return False, (
            f"No students found for subject code {subject_code}"
        )

    for student in students:
        if student[2] != subject_code:
            return False, (
                f"Unexpected subject code found: {student[2]}"
            )

    return True, (
        f"Subject search validation passed for {subject_code}"
    )


def observe_live_endpoints():
    results = []

    try:
        response = requests.get(
            "http://127.0.0.1:5000/students",
            timeout=5
        )
        results.append(f"/students -> HTTP {response.status_code}")
    except Exception as exc:
        results.append(f"/students -> error: {exc}")

    try:
        response = requests.get(
            "http://127.0.0.1:5000/students/by-subject?subject_code=ASD101",
            timeout=5
        )
        results.append(
            f"/students/by-subject -> HTTP {response.status_code}"
        )
    except Exception as exc:
        results.append(f"/students/by-subject -> error: {exc}")

    return results


def call_model(
    model_name,
    system_prompt,
    user_prompt,
    max_tokens=120
):
    try:
        client = OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",
            timeout=180.0
        )

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            max_tokens=max_tokens,
            temperature=0.1
        )

        content = response.choices[0].message.content

        if content and content.strip():
            return content.strip(), None

        return "No response generated.", None

    except Exception as exc:
        return None, (
            f"{model_name} unavailable or timed out ({exc})"
        )


def get_implementation_agent_advice(observe_message):
    prompt = (
        "You are the IMPLEMENTATION AGENT for a Flask "
        "Student Enrolment App.\n\n"

        "Current database fields:\n"
        "- student_id\n"
        "- student_name\n"
        "- subject_code\n\n"

        "Important domain rule:\n"
        "- subject_code is NOT unique.\n"
        "- Multiple students may enrol in the same subject.\n"
        "- Never recommend a unique constraint on subject_code.\n\n"

        "Current endpoints:\n"
        "- GET /students\n"
        "- GET /students/<student_id>\n"
        "- GET /students/by-id\n"
        "- GET /students/by-subject\n"
        "- POST /ask\n\n"

        f"Validation Evidence:\n{observe_message}\n\n"

        "Task:\n"
        "Review ONLY the existing subject-code search feature.\n\n"

        "Check whether the Flask app is running. If it is running, make real "
        "HTTP requests to the relevant endpoints such as /students, "
        "/students/by-id, and /students/by-subject and inspect the actual "
        "responses. Also check the local database. Use that live endpoint "
        "evidence from the running app. Do not rely only on database checks "
        "or code reading. If the app is not responding, say: Unable to "
        "verify live endpoint behavior.\n\n"

        "Rules:\n"
        "- Do not invent new database fields.\n"
        "- Do not invent new endpoints.\n"
        "- Do not modify endpoint contracts.\n"
        "- Do not suggest new application features.\n"
        "- Do not recommend subject_code uniqueness.\n"
        "- Focus only on validation, error handling, "
        "response formatting, or testing.\n"
        "- If the evidence does not support an improvement, "
        "write: No evidence-backed improvement identified.\n"
        "- Return exactly two bullet points, or the no-evidence sentence.\n"
    )

    return call_model(
        IMPLEMENTATION_MODEL,
        (
            "You are a concise implementation assistant. "
            "Follow the rules exactly. "
            "Do not invent requirements."
        ),
        prompt,
        max_tokens=120
    )


def get_review_agent_advice(
    implementation_message,
    observe_message
):
    prompt = (
        "Review ONLY the implementation-agent "
        "recommendation.\n\n"

        f"Implementation Recommendation:\n"
        f"{implementation_message}\n\n"

        f"Validation Evidence:\n"
        f"{observe_message}\n\n"

        "Before reviewing the recommendation, use the validation evidence "
        "provided. Review only the implementation-agent recommendation and "
        "identify evidence-backed risks or corrections. If no evidence-backed "
        "risk exists, say so.\n\n"

        "Application Scope:\n"
        "- database fields: student_id, "
        "student_name, subject_code\n"
        "- endpoints: /students, "
        "/students/<student_id>, "
        "/students/by-id, "
        "/students/by-subject, "
        "/ask\n\n"

        "Important domain rule:\n"
        "- subject_code is NOT unique.\n"
        "- Multiple students may enrol in the same subject.\n"
        "- Any recommendation to make subject_code unique is invalid.\n\n"

        "Rules:\n"
        "- Do not invent new database fields.\n"
        "- Do not invent new endpoints.\n"
        "- Do not suggest new features.\n"
        "- Identify only evidence-backed risks "
        "or corrections.\n"
        "- If no evidence-backed risk exists, say so.\n"
        "- Return exactly three lines.\n\n"

        "Format:\n"
        "Risk: <one short sentence>\n"
        "Correction: <one short sentence>\n"
        "Retest: <one short sentence>\n\n"

        "If no risk is supported by the evidence, use:\n"
        "Risk: No evidence-backed risk identified.\n"
        "Correction: No correction required.\n"
        "Retest: Repeat validation after future changes.\n\n"

        "- Maximum 35 words total.\n"
        "- Do not explain reasoning.\n"
    )

    return call_model(
        REVIEW_MODEL,
        (
            "You are a concise software review assistant. "
            "Follow the output format exactly. "
            "Do not invent requirements."
        ),
        prompt,
        max_tokens=100
    )


def human_review():
    print()
    print("HUMAN REVIEW")
    print("1 - Accept")
    print("2 - Partially Accept")
    print("3 - Reject")

    decision = input("Decision: ").strip()

    if decision == "1":
        return "Accept"

    if decision == "2":
        return "Partially Accept"

    return "Reject"


def adapt(decision):
    print()

    if decision == "Accept":
        print(
            "ADAPT: Apply recommendation and rerun validation."
        )

    elif decision == "Partially Accept":
        print(
            "ADAPT: Apply selected recommendations and "
            "rerun validation."
        )

    else:
        print(
            "ADAPT: Keep current implementation and "
            "document rationale."
        )


def main():
    print("=" * 60)
    print("ASD LAB 02 AGENTIC LOOP")
    print("=" * 60)

    print()
    print("PLAN")
    print(PLAN)

    print()
    print("ACT")
    print("Check local database records")

    ok_data, msg_data = observe_data_quality()

    print()
    print("OBSERVE")
    print(msg_data)

    ok_subject, msg_subject = observe_subject_search(
        "ASD101"
    )

    print(msg_subject)

    live_results = observe_live_endpoints()

    observe_message = (
        f"{msg_data}. "
        f"{msg_subject}. "
        f"Live endpoint checks: " + "; ".join(live_results)
    )

    print()
    print("IMPLEMENTATION AGENT")
    print(f"Model: {IMPLEMENTATION_MODEL}")

    implementation_advice, implementation_error = (
        get_implementation_agent_advice(
            observe_message
        )
    )

    if implementation_advice:
        print()
        print(implementation_advice)
    else:
        print()
        print(implementation_error)
        implementation_advice = (
            "Implementation agent unavailable."
        )

    print()
    print("REVIEW AGENT")
    print(f"Model: {REVIEW_MODEL}")

    review_advice, review_error = (
        get_review_agent_advice(
            implementation_advice,
            observe_message
        )
    )

    if review_advice:
        print()
        print(review_advice)
    else:
        print()
        print(review_error)
    print()
    print("HUMAN DECISION")
    decision = human_review()
    print()
    print(f"Decision: {decision}")
    adapt(decision)
    print()
    print("LOOP COMPLETE")

if __name__ == "__main__":
    main()