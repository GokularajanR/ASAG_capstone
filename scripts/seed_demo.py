"""
Seed script: imports physics_quiz.json and submits sample answers from 6 students.

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
        "name": "Gokularajan R",
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
        "name": "Prashitha JR",
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
        "name": "Ashwin Felix",
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
    {
        "name": "Ayushman Kumar",
        "roll": "PHY2024004",
        "level": "mixed",
        "answers": [
            # Q1 – strong
            "Gravity is a fundamental force of attraction between all objects with mass. On Earth it pulls everything toward the planet's centre, giving objects weight.",
            # Q2 – weak
            "Mass and weight are kind of the same thing, both measure how heavy you are.",
            # Q3 – strong
            "Friction is the resistive force between two surfaces in contact. Without it cars could not brake — brake pads press against the wheel and friction converts kinetic energy to heat, slowing the car.",
            # Q4 – weak
            "Things keep moving or stay still. A force can change that.",
            # Q5 – strong
            "Speed is the scalar measure of how fast an object moves. Velocity is speed with a specified direction, making it a vector quantity. A car doing 60 km/h has speed; 60 km/h due north is velocity.",
            # Q6 – weak
            "Potential energy is stored energy and kinetic energy is moving energy.",
            # Q7 – strong
            "A lever consists of a rigid beam, a fulcrum which is the pivot, the effort force applied by the user, and the load being moved. Placing the fulcrum nearer the load reduces the effort needed to lift it.",
            # Q8 – weak
            "A circuit has wires and a battery. Electricity goes around and powers things.",
            # Q9 – strong
            "Magnets have a north pole and a south pole. Opposite poles attract each other while like poles repel. The magnetic force can act on ferromagnetic materials like iron and nickel without contact.",
            # Q10 – weak
            "F equals ma. More force makes things go faster.",
        ],
    },
    {
        "name": "Yuva Yashvin",
        "roll": "PHY2024005",
        "level": "mixed",
        "answers": [
            # Q1 – weak
            "Gravity pulls things down to Earth.",
            # Q2 – strong
            "Mass is the measure of matter in an object and stays constant everywhere. Weight is the gravitational pull on that mass, so an astronaut weighs less on the Moon but has the same mass.",
            # Q3 – weak
            "Friction happens when things touch each other and slows them down.",
            # Q4 – strong
            "Newton's First Law states that an object remains at rest or in uniform motion in a straight line unless acted upon by an external unbalanced force. This property is called inertia.",
            # Q5 – weak
            "Speed is fast and velocity is also fast but has a direction too.",
            # Q6 – strong
            "Potential energy is energy stored due to an object's position, like a book on a shelf. Kinetic energy is the energy of motion. As the book falls, potential energy converts to kinetic energy.",
            # Q7 – weak
            "A lever helps you lift stuff. There is a pivot in the middle.",
            # Q8 – strong
            "An electric circuit is a complete closed path for current to flow. It requires a source like a battery, conducting wires, and a load such as a bulb. Breaking the circuit stops current flow immediately.",
            # Q9 – weak
            "Magnets have two poles and stick to some metals.",
            # Q10 – strong
            "Newton's Second Law: force equals mass multiplied by acceleration (F = ma). Doubling the force doubles the acceleration. A 5 kg object accelerated at 4 m/s² requires 20 N of net force.",
        ],
    },
    {
        "name": "Ram Tabjulu",
        "roll": "PHY2024006",
        "level": "mixed",
        "answers": [
            # Q1 – average
            "Gravity is a force that pulls objects toward each other. Earth's gravity pulls us down.",
            # Q2 – strong
            "Mass is the amount of matter in an object and does not change with location. Weight is the force of gravity on the object, so it varies — you would weigh less on the Moon than on Earth.",
            # Q3 – average
            "Friction is the force that opposes motion between surfaces. It is useful in brakes and shoes.",
            # Q4 – strong
            "Newton's First Law says an object at rest stays at rest and a moving object keeps moving at constant velocity unless a net external force acts on it. A hockey puck slides because friction is low.",
            # Q5 – average
            "Speed tells you how fast something is moving. Velocity is speed but also includes the direction of movement.",
            # Q6 – weak
            "Potential energy is stored. Kinetic is movement energy.",
            # Q7 – average
            "A lever has a fulcrum, an effort side and a load side. Pushing the effort down lifts the load up.",
            # Q8 – strong
            "A circuit is a closed loop allowing electric current to flow from the positive terminal of a battery through wires and a load like a bulb and back to the negative terminal. An open circuit stops the flow.",
            # Q9 – weak
            "Magnets attract metal objects and have a north and south side.",
            # Q10 – average
            "Newton's Second Law: force equals mass times acceleration. A larger mass requires more force to achieve the same acceleration.",
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
    parser.add_argument("--api", default="https://asag-app.azurewebsites.net")
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
