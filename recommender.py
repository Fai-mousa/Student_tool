"""
student_tool/recommender.py
===========================
Rule-based personalised recommendation engine for the Student Grade Predictor.
No API key or external service required — completely free.

Usage (called from student.py):
    from recommender import get_recommendations
    recs = get_recommendations(student_profile, g3_predicted, total, passed)
    # recs → dict with keys: academic, study, lifestyle, motivation
"""


# ─────────────────────────────────────────────────────────────────────────────
# RULE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def get_recommendations(
    profile: dict,
    g3: float,
    total: float,
    passed: bool,
) -> dict | None:
    """
    Generate personalised recommendations based on the student's profile.

    Parameters
    ----------
    profile : dict   — student input dict (same as raw_input in student.py)
    g3      : float  — predicted G3 (0-50 school scale)
    total   : float  — total score G1+G2+G3
    passed  : bool   — pass/fail verdict

    Returns
    -------
    dict with keys: academic, study, lifestyle, motivation
    Each value is a list of {"icon": str, "tip": str} dicts.
    """

    p           = profile
    studytime   = p.get("studytime", 2)
    failures    = p.get("failures", 0)
    absences    = p.get("absences", 0)
    health      = p.get("health", 4)
    famrel      = p.get("famrel", 4)
    freetime    = p.get("freetime", 3)
    internet    = p.get("internet", "yes")
    higher      = p.get("higher", "yes")
    schoolsup   = p.get("schoolsup", "no")
    famsup      = p.get("famsup", "no")
    activities  = p.get("activities", "no")
    traveltime  = p.get("traveltime", 1)
    g1          = p.get("G1", 0)
    g2          = p.get("G2", 0)
    shortage    = round(60 - total, 1) if not passed else 0

    # ── Grade trend ───────────────────────────────────────────────────────
    grade_improving = g2 > g1
    grade_dropping  = g2 < g1 - 2
    grade_stable    = abs(g2 - g1) <= 2

    # ─────────────────────────────────────────────────────────────────────
    # ACADEMIC TIPS  (focus on grades, failures, support)
    # ─────────────────────────────────────────────────────────────────────
    academic = []

    if failures >= 2:
        academic.append({
            "icon": "🔄",
            "tip": f"With {failures} past failures, early intervention is key — visit your teacher during office hours before the next exam, not after."
        })
    elif failures == 1:
        academic.append({
            "icon": "📌",
            "tip": "You've had one past failure — identify which subject or topic caused it and address it directly with targeted practice."
        })

    if grade_dropping:
        academic.append({
            "icon": "📉",
            "tip": f"Your grade dropped from {g1:.0f} to {g2:.0f} — this trend needs reversing now. Identify what changed between the two periods."
        })
    elif grade_improving:
        academic.append({
            "icon": "📈",
            "tip": f"Great momentum — your grade went from {g1:.0f} to {g2:.0f}. Keep the same habits and push for an even stronger G3."
        })
    elif grade_stable:
        academic.append({
            "icon": "📊",
            "tip": f"Your grades are consistent at around {g2:.0f}/25. A small targeted improvement in weak topics could push your total over the edge."
        })

    if not passed and shortage <= 10:
        academic.append({
            "icon": "🎯",
            "tip": f"You only need {shortage} more points to pass — that's very achievable. Focus on maximising partial marks on every question."
        })
    elif not passed and shortage > 10:
        academic.append({
            "icon": "🏗️",
            "tip": f"You need {shortage} points to pass — start by mastering the core topics worth the most marks in the exam."
        })
    elif passed and g3 >= 40:
        academic.append({
            "icon": "🏆",
            "tip": "You're on track for an excellent grade. Challenge yourself with past exam papers from previous years to aim even higher."
        })
    else:
        academic.append({
            "icon": "✅",
            "tip": "You're predicted to pass — now aim to maximise your score by reviewing any topics you find uncertain."
        })

    if schoolsup == "no" and not passed:
        academic.append({
            "icon": "🏫",
            "tip": "You currently have no school support — ask your teacher about tutoring programmes or study groups available to you."
        })
    elif schoolsup == "yes":
        academic.append({
            "icon": "👨‍🏫",
            "tip": "Make the most of your school support sessions by arriving with specific questions prepared in advance."
        })

    # ─────────────────────────────────────────────────────────────────────
    # STUDY HABIT TIPS  (focus on time, absences, travel, family support)
    # ─────────────────────────────────────────────────────────────────────
    study = []

    if studytime == 1:
        study.append({
            "icon": "⏱️",
            "tip": "You're studying less than 2 hours a week — even adding just 30 minutes daily would significantly change your results."
        })
    elif studytime == 2:
        study.append({
            "icon": "📅",
            "tip": "2–5 hours a week is a reasonable start. Try spreading it across 5 days rather than cramming to improve retention."
        })
    elif studytime == 3:
        study.append({
            "icon": "📚",
            "tip": "5–10 hours a week is solid. Make sure each session has a clear goal — passive re-reading is far less effective than active recall."
        })
    else:
        study.append({
            "icon": "🧠",
            "tip": "You study more than 10 hours a week — make sure quality matches quantity. Use spaced repetition and practice testing for best results."
        })

    if absences > 15:
        study.append({
            "icon": "🚨",
            "tip": f"You have {absences} absences — this is significantly hurting your learning. Each missed class creates a gap that takes double the time to fill."
        })
    elif absences > 6:
        study.append({
            "icon": "📅",
            "tip": f"With {absences} absences, make sure you catch up on every missed lesson the same week — falling behind compounds quickly."
        })
    else:
        study.append({
            "icon": "✅",
            "tip": "Your attendance is good. Consistency in showing up is one of the strongest predictors of academic success — keep it up."
        })

    if traveltime >= 3:
        study.append({
            "icon": "🚌",
            "tip": "Your commute is long — turn travel time into productive time by reviewing notes or listening to educational content on the way."
        })
    elif famsup == "yes":
        study.append({
            "icon": "👨‍👩‍👧",
            "tip": "You have family support — use it actively by asking a family member to quiz you before exams."
        })
    else:
        study.append({
            "icon": "🗓️",
            "tip": "Create a fixed weekly study schedule and treat it like a class — same time, same place builds a powerful habit."
        })

    # ─────────────────────────────────────────────────────────────────────
    # LIFESTYLE TIPS  (health, freetime, internet, activities)
    # ─────────────────────────────────────────────────────────────────────
    lifestyle = []

    if health <= 2:
        lifestyle.append({
            "icon": "❤️",
            "tip": "Your health score is low — poor physical or mental health directly impacts memory and concentration. Speak to a school counsellor if you need support."
        })
    elif health == 3:
        lifestyle.append({
            "icon": "🏃",
            "tip": "Moderate health — even 20 minutes of walking daily has been shown to improve focus and reduce exam stress significantly."
        })
    else:
        lifestyle.append({
            "icon": "💪",
            "tip": "Good health is a real academic asset. Protect your sleep schedule especially in the weeks leading up to exams."
        })

    if famrel <= 2:
        lifestyle.append({
            "icon": "🏠",
            "tip": "A tense home environment makes studying harder. Try to find a quiet space outside the home — a library or café — for focused study sessions."
        })
    elif famrel >= 4:
        lifestyle.append({
            "icon": "🤝",
            "tip": "Good family relationships are a strong support base. Share your academic goals with your family so they can help keep you accountable."
        })
    else:
        lifestyle.append({
            "icon": "🏡",
            "tip": "A stable home environment helps. Keep family and study time well-separated to give full attention to each."
        })

    if freetime >= 4 and not passed:
        lifestyle.append({
            "icon": "⚖️",
            "tip": "You have a lot of free time but are predicted to struggle — consider redirecting just 1–2 hours of it per day toward focused study."
        })
    elif freetime <= 2:
        lifestyle.append({
            "icon": "🧘",
            "tip": "You have very little free time — make sure you schedule proper breaks. Short rest periods actually improve long-term study performance."
        })
    else:
        lifestyle.append({
            "icon": "🎮",
            "tip": "Your free time balance looks reasonable. Use it intentionally — planned leisure is more restorative than unplanned scrolling."
        })

    # ─────────────────────────────────────────────────────────────────────
    # MOTIVATION TIPS  (goal-oriented, verdict-aware)
    # ─────────────────────────────────────────────────────────────────────
    motivation = []

    if higher == "yes":
        motivation.append({
            "icon": "🎓",
            "tip": "You want to pursue higher education — every grade you earn now is a direct step toward that goal. Let your future self pull you forward."
        })
    else:
        motivation.append({
            "icon": "🌍",
            "tip": "Whether or not you pursue higher education, strong academic habits build discipline and problem-solving skills that pay off in every career."
        })

    if passed and g3 >= 40:
        motivation.append({
            "icon": "🌟",
            "tip": "You're on track for an excellent result — excellence becomes a habit, not a one-time event. Stay consistent and don't ease up now."
        })
    elif passed:
        motivation.append({
            "icon": "✨",
            "tip": "You're predicted to pass — that's a real achievement. Use this confidence to push your score even higher in the final stretch."
        })
    else:
        motivation.append({
            "icon": "💡",
            "tip": f"Being predicted to fail is information, not a verdict — you still have time to change this outcome. Focus on the {shortage} points within reach."
        })

    if activities == "yes":
        motivation.append({
            "icon": "🏅",
            "tip": "Extra-curricular involvement shows character — the discipline, teamwork, and resilience you build there directly transfers to academic performance."
        })
    elif internet == "no":
        motivation.append({
            "icon": "📖",
            "tip": "No internet at home? Use school or library resources to access free materials like Khan Academy and BBC Bitesize during breaks."
        })
    else:
        motivation.append({
            "icon": "🌐",
            "tip": "Use your internet access wisely — platforms like Khan Academy, Coursera, and YouTube have free world-class lessons on every subject."
        })

    # ── Trim each category to exactly 3 tips ─────────────────────────────
    return {
        "academic":   academic[:3],
        "study":      study[:3],
        "lifestyle":  lifestyle[:3],
        "motivation": motivation[:3],
    }