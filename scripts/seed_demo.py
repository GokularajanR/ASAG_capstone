"""
Seed script: imports physics_quiz.json and submits sample answers from 3 students.

Usage:
    .venv/Scripts/python.exe scripts/seed_demo.py [--api http://localhost:8000] [--key dev-key]
"""

import argparse
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path

STUDENTS = [
    {
        "name": "Alice Sharma",
        "roll": "PHY2024001",
        "level": "good",
        "answers": [
            # Q1 – What is gravity?
            "Gravity is a force that pulls objects towards each other. On Earth it pulls everything down towards the centre.",
            # Q2 – Mass vs weight
            "Mass is the amount of matter in an object and never changes. Weight is the gravitational force on that object, so it changes on the Moon.",
            # Q3 – Friction + example
            "Friction is the force that resists motion when two surfaces rub together. It is useful in car brakes, which use friction to slow the vehicle down.",
            # Q4 – Newton's 1st Law
            "An object at rest stays at rest and an object in motion continues moving at the same speed and direction unless an unbalanced force acts on it.",
            # Q5 – Speed vs velocity
            "Speed is how fast something moves with no direction. Velocity is speed in a given direction, so it is a vector. 50 km/h is speed; 50 km/h north is velocity.",
            # Q6 – PE vs KE
            "Potential energy is stored energy based on position. Kinetic energy is the energy of a moving object. A ball held high has potential energy; when dropped it converts to kinetic energy.",
            # Q7 – Lever
            "A lever is a simple machine with three parts: the fulcrum is the pivot point, the effort is the force you apply, and the load is what you are lifting. Moving the fulcrum closer to the load makes lifting easier.",
            # Q8 – Electric circuit
            "An electric circuit is a closed loop that allows current to flow. It needs a battery as a power source, wires to carry the current, and a load like a bulb. If the circuit is broken, current stops.",
            # Q9 – Magnets
            "Magnets have two poles, north and south. Opposite poles attract so north and south pull together. Like poles repel so two north poles push apart. They attract iron and steel objects.",
            # Q10 – Newton's 2nd Law
            "Newton's Second Law says F equals m times a. More force means more acceleration. If a 10 kg box is pushed with 30 N, its acceleration is 3 metres per second squared. Greater mass needs more force for the same acceleration.",
        ],
    },
    {
        "name": "Ben Carter",
        "roll": "PHY2024002",
        "level": "average",
        "answers": [
            # Q1
            "Gravity is what makes things fall down. It pulls objects to the ground.",
            # Q2
            "Mass is how heavy something is. Weight depends on gravity so it changes in space.",
            # Q3
            "Friction slows things down when surfaces touch. Brakes use friction to stop cars.",
            # Q4
            "Objects keep doing what they are doing unless a force stops them or moves them.",
            # Q5
            "Speed is how fast you go. Velocity also includes the direction you are moving.",
            # Q6
            "Potential energy is stored like a stretched spring. Kinetic energy is when something is moving. A rollercoaster at the top has potential energy.",
            # Q7
            "A lever helps lift heavy things. You push one end and the other end lifts the load. The middle part is the fulcrum.",
            # Q8
            "A circuit needs a battery and wires and a bulb. The electricity flows around the loop and lights the bulb.",
            # Q9
            "Magnets have north and south poles. Opposite poles attract and same poles repel each other.",
            # Q10
            "Force equals mass times acceleration. A bigger mass needs more force to move it at the same speed.",
        ],
    },
    {
        "name": "Chris Patel",
        "roll": "PHY2024003",
        "level": "weak",
        "answers": [
            # Q1
            "Gravity is the thing that keeps us on the ground.",
            # Q2
            "Mass and weight are both about how heavy you are.",
            # Q3
            "Friction is when things rub. It can help you walk without slipping.",
            # Q4
            "Things stay still unless you push them.",
            # Q5
            "Speed is how fast. Velocity is also fast but different.",
            # Q6
            "Potential energy is energy you have. Kinetic energy is energy when moving.",
            # Q7
            "A lever is a stick that helps lift things. You push down and the other side goes up.",
            # Q8
            "A circuit is wires and a battery that makes electricity go to a bulb.",
            # Q9
            "Magnets stick to metal things and have two sides.",
            # Q10
            "Force is mass times acceleration. More mass means harder to push.",
        ],
    },
]


def api(url, data=None, key="dev-key"):
    body = json.dumps(data).encode() if data else None
    headers = {"Content-Type": "application/json", "X-API-Key": key}
    req = urllib.request.Request(url, data=body, headers=headers,
                                 method="POST" if data else "GET")
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--key", default="dev-key")
    parser.add_argument("--quiz", default="physics_quiz.json")
    args = parser.parse_args()

    base = args.api.rstrip("/")

    # 1 – load quiz file
    quiz_path = Path(args.quiz)
    if not quiz_path.exists():
        sys.exit(f"Quiz file not found: {quiz_path}")
    quiz = json.loads(quiz_path.read_text())
    raw_questions = quiz.get("questions", quiz)

    # 2 – import questions
    print(f"Importing {len(raw_questions)} questions…")
    question_ids = []
    for q in raw_questions:
        rec = api(f"{base}/questions", {
            "text": q["text"],
            "reference_answer": q["reference_answer"],
            "subject": q.get("subject", ""),
            "max_marks": q.get("max_marks", 5),
        }, key=args.key)
        question_ids.append(rec["id"])
        print(f"  + [{rec['id'][:8]}] {q['text'][:60]}")

    # 3 – submit student answers
    print(f"\nSubmitting answers for {len(STUDENTS)} students…")
    for student in STUDENTS:
        print(f"\n  {student['name']} ({student['roll']}) — {student['level']}")
        for qid, answer in zip(question_ids, student["answers"]):
            rec = api(f"{base}/submissions", {
                "question_id": qid,
                "student_name": student["name"],
                "roll_number": student["roll"],
                "answer": answer,
            }, key=args.key)
            print(f"    submitted [{qid[:8]}]")

    print(f"\nDone. Open the teacher dashboard to grade submissions.")
    print(f"  {base.replace('localhost', '127.0.0.1')}")


if __name__ == "__main__":
    main()
