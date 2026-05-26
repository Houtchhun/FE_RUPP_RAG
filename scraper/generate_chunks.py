#!/usr/bin/env python3
"""Generate chunks from all_programs_comprehensive.json."""

import json
from pathlib import Path

DATA_PATH = Path(__file__).parent / "outputs" / "all_programs_comprehensive.json"
OUTPUT_DIR = Path(__file__).parent / "outputs" / "chunks"


def _format_curriculum(prog_name: str, curriculum: dict) -> str:
    """Return a formatted curriculum string, handling foundation/year_X keys and semester_2_tracks."""
    text = f"{prog_name} - Curriculum:\n"
    note = curriculum.get("note")
    if note:
        text += f"({note})\n"

    for year_key, semesters in curriculum.items():
        if year_key == "note":
            continue
        label = "Foundation Year" if year_key == "foundation" else f"Year {year_key.split('_')[1]}"
        text += f"\n{label}:\n"

        for sem_key, value in semesters.items():
            if sem_key == "semester_2_tracks":
                text += "  Semester 2 (choose one track):\n"
                for track_name, courses in value.items():
                    track_label = track_name.replace("_", " ").title()
                    text += f"    [{track_label}]\n"
                    for c in courses:
                        credit = c.get("credit") or "N/A"
                        text += f"      - {c['subject']} ({credit} credits)\n"
            else:
                sem_num = sem_key.split("_")[1]
                text += f"  Semester {sem_num}:\n"
                for c in value:
                    credit = c.get("credit") or "N/A"
                    text += f"    - {c['subject']} ({credit} credits)\n"
    return text


def _chunks_for_program(prog: dict, category: str) -> list:
    chunks = []
    pid = prog["id"]
    name = prog["name"]
    prefix = "program" if category != "master" else "master"

    # 1. Overview chunk
    overview = f"{name} ({category.replace('_', ' ').title()})\n\n"
    overview += prog.get("description", "") + "\n\n"
    overview += prog.get("overview", "")
    if prog.get("established"):
        overview += f"\n\nEstablished: {prog['established']}"
    chunks.append({
        "chunk_id": f"{pid}_overview",
        "text": overview.strip(),
        "metadata": {"url": f"{prefix}:{pid}", "title": name, "type": "program_overview", "category": category}
    })

    # 2. Details summary chunk
    details = f"{name} - Program Details:\n"
    details += f"Category: {category.replace('_', ' ').title()}\n"
    details += f"Degree: {prog.get('degree_type', 'N/A')}\n"
    details += f"Duration: {prog.get('duration_years', 'N/A')} years"
    if prog.get("duration_semesters"):
        details += f" ({prog['duration_semesters']} semesters)"
    details += "\n"
    if prog.get("max_study_duration_years"):
        details += f"Maximum Study Duration: {prog['max_study_duration_years']} years\n"
    details += f"Language: {prog.get('language', 'English')}\n"
    established = prog.get("established") or (str(prog["established_year"]) if prog.get("established_year") else None)
    if established:
        details += f"Established: {established}\n"
    if prog.get("gpa_requirement"):
        details += f"GPA Requirement: {prog['gpa_requirement']}\n"
    if prog.get("thesis_gpa_requirement"):
        details += f"Thesis GPA Requirement: {prog['thesis_gpa_requirement']}\n"
    if prog.get("total_credits"):
        details += f"Total Credits: {prog['total_credits']}\n"
    if prog.get("thesis_required") is not None:
        details += f"Thesis Required: {'Yes' if prog['thesis_required'] else 'No (multiple track options)'}\n"
    chunks.append({
        "chunk_id": f"{pid}_details",
        "text": details.strip(),
        "metadata": {"url": f"{prefix}:{pid}", "title": name, "type": "program_details", "category": category}
    })

    # 3. Curriculum chunk(s)
    if prog.get("curriculum"):
        curr = prog["curriculum"]
        curriculum_text = _format_curriculum(name, curr)
        if prog.get("curriculum_note"):
            curriculum_text += f"\nNote: {prog['curriculum_note']}"
        chunks.append({
            "chunk_id": f"{pid}_curriculum",
            "text": curriculum_text,
            "metadata": {"url": f"{prefix}:{pid}", "title": name, "type": "curriculum", "category": category}
        })

        # Per-year chunks for better retrieval granularity
        for year_key, semesters in curr.items():
            if year_key == "note":
                continue
            label = "Foundation Year" if year_key == "foundation" else f"Year {year_key.split('_')[1]}"
            year_text = f"{name} - {label} Courses:\n"
            for sem_key, value in semesters.items():
                if sem_key == "semester_2_tracks":
                    year_text += "Semester 2 (choose one track):\n"
                    for track_name, courses in value.items():
                        track_label = track_name.replace("_", " ").title()
                        year_text += f"  [{track_label}]\n"
                        for c in courses:
                            credit = c.get("credit") or "N/A"
                            year_text += f"    - {c['subject']} ({credit} credits)\n"
                else:
                    sem_num = sem_key.split("_")[1]
                    year_text += f"Semester {sem_num}:\n"
                    for c in value:
                        credit = c.get("credit") or "N/A"
                        year_text += f"  - {c['subject']} ({credit} credits)\n"
            chunks.append({
                "chunk_id": f"{pid}_curriculum_{year_key}",
                "text": year_text.strip(),
                "metadata": {"url": f"{prefix}:{pid}", "title": name, "type": "curriculum_year", "category": category}
            })

    # 4. Job/career prospects chunk (handles both "job_prospects" and "career_prospects" keys)
    prospects = prog.get("job_prospects") or prog.get("career_prospects")
    if prospects:
        text = f"{name} - Career Prospects:\n" + "\n".join(f"- {j}" for j in prospects)
        chunks.append({
            "chunk_id": f"{pid}_careers",
            "text": text,
            "metadata": {"url": f"{prefix}:{pid}", "title": name, "type": "careers", "category": category}
        })

    # 5. Research areas / focus chunk (handles research_areas, research_focus, research_focus_areas)
    for field in ("research_areas", "research_focus", "research_focus_areas"):
        if prog.get(field):
            label = "Research Focus Areas" if field == "research_focus_areas" else ("Research Areas" if field == "research_areas" else "Research Focus")
            text = f"{name} - {label}:\n" + "\n".join(f"- {a}" for a in prog[field])
            chunks.append({
                "chunk_id": f"{pid}_research",
                "text": text,
                "metadata": {"url": f"{prefix}:{pid}", "title": name, "type": "research", "category": category}
            })
            break

    # 6. Partnerships chunk
    if prog.get("partnerships"):
        text = f"{name} - Partnerships:\n" + "\n".join(f"- {p}" for p in prog["partnerships"])
        chunks.append({
            "chunk_id": f"{pid}_partnerships",
            "text": text,
            "metadata": {"url": f"{prefix}:{pid}", "title": name, "type": "partnerships", "category": category}
        })

    # 7. Final semester optional tracks
    if prog.get("last_semester_optional_tracks"):
        text = f"{name} - Final Semester Track Options:\n"
        text += "\n".join(f"- {t}" for t in prog["last_semester_optional_tracks"])
        chunks.append({
            "chunk_id": f"{pid}_tracks",
            "text": text,
            "metadata": {"url": f"{prefix}:{pid}", "title": name, "type": "program_tracks", "category": category}
        })

    # 8. Admission requirements chunk (master programs)
    if prog.get("admission_requirements"):
        reqs = prog["admission_requirements"]
        text = f"{name} - Admission Requirements:\n"
        if reqs.get("bachelor_degree"):
            text += f"Bachelor Degree: {reqs['bachelor_degree']}\n"
        if reqs.get("english_proficiency"):
            ep = reqs["english_proficiency"]
            text += "English Proficiency:\n"
            if isinstance(ep, dict):
                if ep.get("minimum_score"):
                    text += f"  Minimum Score: {ep['minimum_score']}\n"
                if ep.get("accepted_tests"):
                    for t in ep["accepted_tests"]:
                        text += f"  - {t}\n"
            elif isinstance(ep, list):
                for item in ep:
                    text += f"  - {item}\n"
        if reqs.get("application_documents"):
            text += "Required Documents:\n"
            for doc in reqs["application_documents"]:
                text += f"  - {doc}\n"
        if reqs.get("important_notes"):
            text += "Important Notes:\n"
            for note in reqs["important_notes"]:
                text += f"  - {note}\n"
        chunks.append({
            "chunk_id": f"{pid}_admission",
            "text": text.strip(),
            "metadata": {"url": f"{prefix}:{pid}", "title": name, "type": "admission", "category": category}
        })

    # 9. Vision / objectives / benefits (master programs)
    vision_parts = []
    if prog.get("department_vision"):
        vision_parts.append(f"Vision: {prog['department_vision']}")
    if prog.get("department_mission"):
        vision_parts.append("Mission:\n" + "\n".join(f"  - {m}" for m in prog["department_mission"]))
    if prog.get("strategic_objective"):
        vision_parts.append(f"Strategic Objective: {prog['strategic_objective']}")
    if prog.get("mission"):
        vision_parts.append(f"Mission: {prog['mission']}")
    if prog.get("benefits"):
        vision_parts.append("Benefits:\n" + "\n".join(f"  - {b}" for b in prog["benefits"]))
    if vision_parts:
        text = f"{name} - Vision & Objectives:\n" + "\n".join(vision_parts)
        chunks.append({
            "chunk_id": f"{pid}_vision",
            "text": text.strip(),
            "metadata": {"url": f"{prefix}:{pid}", "title": name, "type": "vision_objectives", "category": category}
        })

    # 10. Program tracks (master)
    if prog.get("program_tracks"):
        text = f"{name} - Program Tracks:\n" + "\n".join(f"- {t}" for t in prog["program_tracks"])
        chunks.append({
            "chunk_id": f"{pid}_program_tracks",
            "text": text,
            "metadata": {"url": f"{prefix}:{pid}", "title": name, "type": "program_tracks", "category": category}
        })

    # 11. Internship requirement
    if prog.get("internship_requirement"):
        text = f"{name} - Internship Requirement:\n{prog['internship_requirement']}"
        chunks.append({
            "chunk_id": f"{pid}_internship",
            "text": text,
            "metadata": {"url": f"{prefix}:{pid}", "title": name, "type": "internship", "category": category}
        })

    # 12. Degree requirements
    if prog.get("degree_requirements"):
        text = f"{name} - Degree Requirements:\n" + "\n".join(f"- {r}" for r in prog["degree_requirements"])
        chunks.append({
            "chunk_id": f"{pid}_degree_requirements",
            "text": text,
            "metadata": {"url": f"{prefix}:{pid}", "title": name, "type": "degree_requirements", "category": category}
        })

    # 13. 2+2 transfer note
    if prog.get("program_2plus2_note"):
        text = f"{name} - 2+2 Transfer Program:\n{prog['program_2plus2_note']}"
        chunks.append({
            "chunk_id": f"{pid}_2plus2",
            "text": text,
            "metadata": {"url": f"{prefix}:{pid}", "title": name, "type": "transfer_program", "category": category}
        })

    # 14. Program features
    if prog.get("program_features"):
        text = f"{name} - Program Features:\n" + "\n".join(f"- {feat}" for feat in prog["program_features"])
        chunks.append({
            "chunk_id": f"{pid}_features",
            "text": text,
            "metadata": {"url": f"{prefix}:{pid}", "title": name, "type": "program_features", "category": category}
        })

    # 15. Specialization tracks (with descriptions)
    if prog.get("specialization_tracks"):
        text = f"{name} - Specialization Tracks:\n"
        for track in prog["specialization_tracks"]:
            text += f"\n{track['name']}:\n  {track.get('description', '')}\n"
        chunks.append({
            "chunk_id": f"{pid}_specialization_tracks",
            "text": text.strip(),
            "metadata": {"url": f"{prefix}:{pid}", "title": name, "type": "specialization_tracks", "category": category}
        })

    # 16. Cooperation efforts
    if prog.get("cooperation_efforts"):
        text = f"{name} - International Cooperation:\n" + "\n".join(f"- {c}" for c in prog["cooperation_efforts"])
        chunks.append({
            "chunk_id": f"{pid}_cooperation",
            "text": text,
            "metadata": {"url": f"{prefix}:{pid}", "title": name, "type": "cooperation", "category": category}
        })

    return chunks


def generate_chunks():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    all_chunks = []

    # Institution-level chunk
    inst_text = (
        f"Institution: {data['institution']}\n"
        f"Total Programs: {data['total_programs']}\n"
        f"  - Honor Bachelor Programs: {data['programs_summary']['honor_bachelor_count']}\n"
        f"  - Bachelor Programs: {data['programs_summary']['bachelor_count']}\n"
        f"  - Master Programs: {data['programs_summary']['master_count']}\n"
    )
    all_chunks.append({
        "chunk_id": "institution_overview",
        "text": inst_text.strip(),
        "metadata": {"url": "institution", "title": data["institution"], "type": "institution_overview"}
    })

    # Programs list chunk
    prog_list = (
        f"Academic Programs offered at {data['institution']}:\n"
        f"FE-RUPP offers a total of {data['total_programs']} programs across three categories.\n\n"
        f"HONOR BACHELOR PROGRAMS ({data['programs_summary']['honor_bachelor_count']} programs):\n"
    )
    for p in data.get("honor_bachelor_programs", []):
        prog_list += f"  - {p['name']} ({p.get('duration_years', 4)} years)\n"
    prog_list += f"\nBACHELOR PROGRAMS ({data['programs_summary']['bachelor_count']} programs):\n"
    for p in data.get("bachelor_programs", []):
        prog_list += f"  - {p['name']} ({p.get('duration_years', 4)} years)\n"
    prog_list += f"\nMASTER PROGRAMS ({data['programs_summary']['master_count']} programs):\n"
    for p in data.get("master_programs", []):
        prog_list += f"  - {p['name']} ({p.get('duration_years', 'N/A')} years)\n"
    prog_list += (
        "\nTo learn more about any specific program, ask about program details, "
        "curriculum, admission requirements, or career prospects."
    )
    all_chunks.append({
        "chunk_id": "programs_list",
        "text": prog_list.strip(),
        "metadata": {"url": "institution", "title": "Programs List", "type": "programs_list"}
    })

    # Grading scale chunk
    gs = data.get("grading_scale", {})
    if gs:
        gs_text = f"FE-RUPP Grading Scale ({gs.get('scale', '4.00 GPA')}):\n"
        for g in gs.get("grades", []):
            gs_text += f"  {g['score_range']}% → {g['grade']} (GPA {g['gpa']}) - {g['description']}\n"
        if gs.get("note"):
            gs_text += f"\nNote: {gs['note']}"
        all_chunks.append({
            "chunk_id": "grading_scale",
            "text": gs_text.strip(),
            "metadata": {"url": "institution", "title": "Grading Scale", "type": "grading_scale"}
        })

    # General info chunk
    gi = data.get("general_info", {})
    if gi:
        gi_text = "FE-RUPP General Information:\n"
        if gi.get("admission_requirements"):
            gi_text += f"Admission Requirements: {gi['admission_requirements']}\n"
        if gi.get("scholarship_info"):
            gi_text += f"Scholarship: {gi['scholarship_info']}\n"
        if gi.get("internship_requirement"):
            gi_text += f"Internship: {gi['internship_requirement']}\n"
        if gi.get("contact"):
            c = gi["contact"]
            gi_text += f"Contact: {c.get('address', '')} | Tel: {c.get('phone', '')} | Email: {c.get('email', '')}\n"
        all_chunks.append({
            "chunk_id": "general_info",
            "text": gi_text.strip(),
            "metadata": {"url": "institution", "title": "General Information", "type": "general_info"}
        })

    # About Us chunks
    au = data.get("about_us", {})
    if au:
        # Vision & Mission chunk
        vm_text = "FE-RUPP Vision:\n" + au.get("vision", "") + "\n\nFE-RUPP Mission:\n"
        vm_text += "\n".join(f"• {m}" for m in au.get("mission", []))
        all_chunks.append({
            "chunk_id": "about_vision_mission",
            "text": vm_text.strip(),
            "metadata": {"url": "about", "title": "FE-RUPP Vision & Mission", "type": "vision_mission"}
        })

        # Dean's message chunk
        dean = au.get("dean_message", {})
        if dean:
            dean_text = (
                f"Dean's Message - FE-RUPP\n"
                f"{dean.get('dean_name', '')}, {dean.get('title', '')}\n\n"
                f"{dean.get('message', '')}\n\n"
                f"Teaching Framework: {dean.get('teaching_framework', '')}\n"
                f"Faculty Established: {dean.get('established', '')}"
            )
            all_chunks.append({
                "chunk_id": "about_dean_message",
                "text": dean_text.strip(),
                "metadata": {"url": "about", "title": "Dean's Message", "type": "dean_message"}
            })

        # Departments & skills chunk
        depts = au.get("departments", [])
        if depts:
            dept_text = "FE-RUPP Departments and Graduate Skills:\n"
            for dept in depts:
                dept_text += f"\n{dept['name']}:\n"
                dept_text += "\n".join(f"  - {s}" for s in dept.get("main_skills", []))
                dept_text += "\n"
            all_chunks.append({
                "chunk_id": "about_departments_skills",
                "text": dept_text.strip(),
                "metadata": {"url": "about", "title": "Departments and Skills", "type": "departments_skills"}
            })

        # Soft skills program chunk
        fs = au.get("faculty_skills", {})
        soft = fs.get("soft_skills", [])
        soft_prog = fs.get("soft_skills_program", {})
        if soft:
            soft_text = "FE-RUPP Soft Skills Program:\n"
            soft_text += "\n".join(f"- {s}" for s in soft)
            if soft_prog:
                soft_text += f"\n\nStudy: {soft_prog.get('study_hours', '')}"
                soft_text += f"\nAward: {soft_prog.get('award', '')}"
            all_chunks.append({
                "chunk_id": "about_soft_skills",
                "text": soft_text.strip(),
                "metadata": {"url": "about", "title": "Soft Skills Program", "type": "soft_skills"}
            })

        # Contact / location chunk
        contact = au.get("contact", {})
        if contact:
            contact_text = (
                f"FE-RUPP Contact Information:\n"
                f"Office: {contact.get('office', '')}\n"
                f"Location: {contact.get('location', '')}\n"
                f"Email: {contact.get('email', '')}\n"
                f"Phone: {contact.get('phone', '')}"
            )
            all_chunks.append({
                "chunk_id": "about_contact",
                "text": contact_text.strip(),
                "metadata": {"url": "about", "title": "Contact Information", "type": "contact"}
            })

    # Per-program chunks
    for prog in data.get("honor_bachelor_programs", []):
        all_chunks.extend(_chunks_for_program(prog, "honor_bachelor"))

    for prog in data.get("bachelor_programs", []):
        all_chunks.extend(_chunks_for_program(prog, "bachelor"))

    for prog in data.get("master_programs", []):
        all_chunks.extend(_chunks_for_program(prog, "master"))

    # Save
    output_file = OUTPUT_DIR / "all_chunks.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"[OK] Created {len(all_chunks)} chunks")
    print(f"[OK] Saved to {output_file}")
    return len(all_chunks)


if __name__ == "__main__":
    generate_chunks()
