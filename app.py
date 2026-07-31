from flask import Flask, render_template, request, send_from_directory
from dotenv import load_dotenv
from openai import OpenAI
import sqlite3
import os

load_dotenv()

DATABASE_NAME = "enrolment.db"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")

app = Flask(__name__)

client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama"
)


def get_db_connection():
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    return conn


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/css/<path:filename>")
def css(filename):
    return send_from_directory("css", filename)


@app.route("/students")
def get_students():
    conn = get_db_connection()
    students = conn.execute(
        "SELECT student_id, student_name, subject_code FROM students"
    ).fetchall()
    conn.close()

    html = "<table>"
    html += (
        "<thead><tr>"
        "<th>ID</th><th>Name</th><th>Subject</th>"
        "</tr></thead>"
    )
    html += "<tbody>"
    for student in students:
        html += (
            f"<tr>"
            f"<td>{student['student_id']}</td>"
            f"<td>{student['student_name']}</td>"
            f"<td>{student['subject_code']}</td>"
            f"</tr>"
        )
    html += "</tbody></table>"

    return html

@app.route("/students/<int:student_id>")
def get_student(student_id):
    conn = get_db_connection()
    student = conn.execute(
        "SELECT student_id, student_name, subject_code FROM students WHERE student_id = ?",
        (student_id,)
    ).fetchone()
    conn.close()

    if student is None:
        return (
            f"<p>❌ Student with ID <strong>{student_id}</strong> "
            f"was not found. Please check the ID and try again.</p>"
        ), 200

    return (
        f"<p>"
        f"ID: {student['student_id']}<br>"
        f"Name: {student['student_name']}<br>"
        f"Subject: {student['subject_code']}"
        f"</p>"
    )


@app.route("/students/by-id")
def get_student_by_id():
    student_id_raw = request.args.get("student_id", "").strip()

    if not student_id_raw:
        return "<p>Student ID is required.</p>", 400

    if not student_id_raw.isdigit():
        return "<p>Student ID must be a positive integer.</p>", 400

    return get_student(int(student_id_raw))


@app.route("/ask", methods=["POST"])
def ask_local_agent():
    question = request.form.get("question", "").strip()

    if not question:
        return "<p>Question is required.</p>", 400

    try:
        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise software engineering assistant. "
                        "Answer in one short paragraph unless asked otherwise."
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

        answer = response.choices[0].message.content

        return f"<p>{answer}</p>"

    except Exception as exc:
        return (
            "<p>Local AI agent request failed. "
            "Check that Ollama is running and that qwen2.5:0.5b is installed.</p>"
            f"<pre>{exc}</pre>",
            503,
        )


if __name__ == "__main__":
    app.run(debug=True, port=5001)