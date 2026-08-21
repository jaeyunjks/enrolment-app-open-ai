from flask import Flask, render_template, request, send_from_directory
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
import sqlite3
import os

load_dotenv()

DATABASE_NAME = "enrolment.db"

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434/v1"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5:0.5b"
)

app = Flask(__name__)

client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama"
)

# ============================== Prompt Assets ==============================

PROMPT_DIR = Path(__file__).with_name("prompts")


def load_prompt(filename):
    prompt_path = PROMPT_DIR / filename
    return prompt_path.read_text(
        encoding="utf-8"
    ).strip()


# ============================== Database ==============================

def get_db_connection():
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    return conn


# ============================== Main Routes ==============================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/css/<path:filename>")
def css(filename):
    return send_from_directory("css", filename)


# ============================== Student Routes ==============================

@app.route("/students")
def get_students():
    conn = get_db_connection()

    students = conn.execute(
        """
        SELECT
            student_id,
            student_name,
            subject_code
        FROM students
        """
    ).fetchall()

    conn.close()

    html = "<ul>"

    for student in students:
        html += (
            f"<li>"
            f"{student['student_id']} - "
            f"{student['student_name']} - "
            f"{student['subject_code']}"
            f"</li>"
        )

    html += "</ul>"

    return html


@app.route("/students/<int:student_id>")
def get_student(student_id):
    conn = get_db_connection()

    student = conn.execute(
        """
        SELECT
            student_id,
            student_name,
            subject_code
        FROM students
        WHERE student_id = ?
        """,
        (student_id,)
    ).fetchone()

    conn.close()

    if student is None:
        return "<p>Student not found.</p>", 404

    return (
        f"<p>"
        f"ID: {student['student_id']}<br>"
        f"Name: {student['student_name']}<br>"
        f"Subject: {student['subject_code']}"
        f"</p>"
    )


@app.route("/students/by-id")
def get_student_by_id():
    student_id_raw = request.args.get(
        "student_id",
        ""
    ).strip()

    if not student_id_raw:
        return "<p>Student ID is required.</p>", 400

    if not student_id_raw.isdigit():
        return (
            "<p>Student ID must be a positive integer.</p>",
            400,
        )

    return get_student(
        int(student_id_raw)
    )


@app.route("/students/by-subject")
def get_students_by_subject():
    subject_code = request.args.get(
        "subject_code",
        ""
    ).strip().upper()

    if not subject_code:
        return "<p>Subject code is required.</p>", 400

    # Week 2 evidence-backed improvement:
    # malformed subject codes return HTTP 400,
    # while valid-but-not-found codes return HTTP 404.
    if (
        not subject_code.isalnum()
        or len(subject_code) > 10
    ):
        return (
            "<p>Invalid subject code format.</p>",
            400,
        )

    conn = get_db_connection()

    students = conn.execute(
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
        return (
            f"<p>No students found for subject code "
            f"{subject_code}.</p>",
            404,
        )

    html = "<ul>"

    for student in students:
        html += (
            f"<li>"
            f"{student['student_id']} - "
            f"{student['student_name']} - "
            f"{student['subject_code']}"
            f"</li>"
        )

    html += "</ul>"

    return html


# ============================== Local AI Endpoint ==============================

@app.route("/ask", methods=["POST"])
def ask_local_agent():
    question = request.form.get(
        "question",
        ""
    ).strip()

    if not question:
        return "<p>Question is required.</p>", 400

    try:
        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise software "
                        "engineering assistant. "
                        "Answer in one short paragraph "
                        "unless asked otherwise."
                    )
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            max_tokens=200,
            temperature=0.2,
        )

        answer = (
            response
            .choices[0]
            .message
            .content
        )

        return f"<p>{answer}</p>"

    except Exception as exc:
        return (
            "<p>Local AI agent request failed. "
            "Check that Ollama is running and that "
            "qwen2.5:0.5b is installed.</p>"
            f"<pre>{exc}</pre>",
            503,
        )


# ============================== Context-Aware AI Endpoint ==============================

@app.route("/ask-with-context", methods=["POST"])
def ask_with_context():
    question = request.form.get(
        "question",
        ""
    ).strip()

    if not question:
        return "<p>Question is required.</p>", 400

    try:
        # Load reusable Week 3 prompt assets.
        system_prompt = load_prompt(
            "implementation_system_prompt.txt"
        )

        task_prompt = load_prompt(
            "context_qa_task_prompt.txt"
        )

        # Runtime prompt assembly:
        # reusable task specification + current user input.
        final_prompt = (
            f"{task_prompt}\n\n"
            f"User Question:\n"
            f"{question}"
        )

        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": final_prompt
                }
            ],
            max_tokens=300,
            temperature=0,
        )

        answer = (
            response
            .choices[0]
            .message
            .content
        )

        return f"<p>{answer}</p>"

    except Exception as exc:
        return (
            "<p>Context-aware request failed.</p>"
            f"<pre>{exc}</pre>",
            503,
        )


# ============================== Run App ==============================

if __name__ == "__main__":
    app.run(debug=True)