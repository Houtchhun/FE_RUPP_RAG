import json
import shutil
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


CHUNKS_PATH       = Path(__file__).parent.parent / "scraper" / "outputs" / "chunks" / "all_chunks.json"
COMPREHENSIVE_PATH = Path(__file__).parent.parent / "scraper" / "outputs" / "all_programs_comprehensive.json"
DB_DIR            = Path(__file__).parent / "chroma_db"

EMBEDDING_MODEL   = "BAAI/bge-base-en-v1.5"


# ── Metadata helpers ──────────────────────────────────────────────────────────

def _clean_metadata(raw: dict) -> dict:
    """Flatten metadata so every value is a plain string/int/float for ChromaDB."""
    clean = {}
    for k, v in raw.items():
        if v is None:
            continue
        clean[k] = ", ".join(str(x) for x in v) if isinstance(v, list) else str(v)
    return clean


# ── Curriculum helpers ────────────────────────────────────────────────────────

def _year_label(year_key: str) -> str:
    if year_key == "foundation":
        return "Foundation Year"
    num = year_key.split("_")[1]
    return f"Year {num}"


def _year_db_term(year_key: str) -> str:
    """Returns the suffix that _YEAR_TO_DB_TERM maps to — used as chunk title suffix."""
    if year_key == "foundation":
        return "Foundation Year Courses Semester"
    num = year_key.split("_")[1]
    return f"Year {num} Courses Semester"


def _year_meta(year_key: str) -> str:
    """Canonical metadata value stored in ChromaDB 'year' field."""
    if year_key == "foundation":
        return "foundation"
    return year_key   # e.g. "year_2", "year_3", "year_4", "year_1"


def _format_semester_block(sem_key: str, value) -> str:
    """Format a single semester or track block into text."""
    if sem_key == "semester_2_tracks":
        lines = ["\nSemester 2 (choose one track):"]
        for track_name, courses in value.items():
            track_label = track_name.replace("_", " ").title()
            lines.append(f"  [{track_label}]")
            for c in courses:
                credit = c.get("credit") or "N/A"
                lines.append(f"    - {c['subject']} ({credit} credits)")
        return "\n".join(lines)
    sem_num = sem_key.split("_")[1]
    lines = [f"\nSemester {sem_num}:"]
    for c in value:
        credit = c.get("credit") or "N/A"
        lines.append(f"  - {c['subject']} ({credit} credits)")
    return "\n".join(lines)


def _docs_for_curriculum_by_year(prog: dict, source_prefix: str, pid: str, category: str) -> list:
    """One LangChain Document per curriculum year, with 'year' metadata for filtering."""
    name       = prog["name"]
    curriculum = prog.get("curriculum", {})
    docs       = []

    for year_key, semesters in curriculum.items():
        if year_key == "note":
            continue

        db_term  = _year_db_term(year_key)
        label    = _year_label(year_key)
        year_tag = _year_meta(year_key)

        # Title matches the normalised query string exactly (BM25 + vector both benefit)
        lines = [
            f"{name} {db_term}",
            "",
            f"{name} — {label} Courses:",
        ]
        for sem_key, value in semesters.items():
            lines.append(_format_semester_block(sem_key, value))

        text = "\n".join(lines)

        docs.append(Document(
            page_content=text,
            metadata={
                "source":       f"{source_prefix}:{pid}:curriculum:{year_tag}",
                "type":         "curriculum",
                "program_name": name,
                "category":     category,
                "year":         year_tag,
            },
        ))

    return docs


# ── Per-program document factory ──────────────────────────────────────────────

def _docs_for_program(prog: dict, category: str) -> list:
    """Convert a program entry to LangChain Documents."""
    pid           = prog["id"]
    name          = prog["name"]
    source_prefix = "master_program" if category == "master" else "program"
    docs          = []

    # Overview document
    degree   = prog.get("degree_type", "N/A")
    duration = prog.get("duration_years", "N/A")
    language = prog.get("language", "English")
    summary  = (
        f"The {name} program awards a {degree} degree "
        f"with a duration of {duration} years, taught in {language}.\n\n"
    )
    overview_text = summary + (
        f"{'Master ' if category == 'master' else ''}Program: {name}\n"
        f"Category: {category.replace('_', ' ').title()}\n"
        f"Degree: {degree}\n"
        f"Duration: {duration} years"
    )
    if prog.get("duration_semesters"):
        overview_text += f" ({prog['duration_semesters']} semesters)"
    overview_text += f"\nLanguage: {language}\n"
    overview_text += f"\nDescription: {prog.get('description', '')}\n"
    if prog.get("overview"):
        overview_text += f"\nOverview: {prog['overview']}\n"
    if prog.get("partnerships"):
        overview_text += f"\nPartnerships: {', '.join(prog['partnerships'])}\n"
    if prog.get("established"):
        overview_text += f"Established: {prog['established']}\n"

    docs.append(Document(
        page_content=overview_text,
        metadata={
            "source":       f"{source_prefix}:{pid}",
            "type":         f"{source_prefix}_overview",
            "program_name": name,
            "category":     category,
        },
    ))

    # Curriculum — one Document per year (with year metadata for filtered retrieval)
    if prog.get("curriculum"):
        docs.extend(_docs_for_curriculum_by_year(prog, source_prefix, pid, category))

    # Job prospects
    if prog.get("job_prospects"):
        job_text = (
            f"Career Prospects for {name} Graduates:\n\n"
            + "\n".join(f"- {job}" for job in prog["job_prospects"])
        )
        docs.append(Document(
            page_content=job_text,
            metadata={
                "source":       f"{source_prefix}:{pid}:careers",
                "type":         "careers",
                "program_name": name,
                "category":     category,
            },
        ))

    # Research areas / focus
    for field in ("research_areas", "research_focus"):
        if prog.get(field):
            label        = "Research Areas" if field == "research_areas" else "Research Focus"
            research_text = (
                f"{label} for {name}:\n\n"
                + "\n".join(f"- {area}" for area in prog[field])
            )
            docs.append(Document(
                page_content=research_text,
                metadata={
                    "source":       f"{source_prefix}:{pid}:research",
                    "type":         "research",
                    "program_name": name,
                    "category":     category,
                },
            ))
            break

    # Thesis / GPA requirement
    if prog.get("thesis_gpa_requirement"):
        gpa      = prog["thesis_gpa_requirement"]
        gpa_text = (
            f"Thesis and Graduation Requirements for {name}:\n\n"
            f"Students in {name} need a minimum GPA of {gpa} "
            f"(GPA > {gpa}) to be eligible to write a research thesis.\n"
            f"The minimum GPA required to write a thesis in {name} is {gpa}.\n\n"
            f"Final semester options for {name} students:\n"
        )
        for track in prog.get("last_semester_optional_tracks", []):
            gpa_text += f"  - {track}\n"
        docs.append(Document(
            page_content=gpa_text,
            metadata={
                "source":       f"{source_prefix}:{pid}:graduation",
                "type":         "graduation_requirements",
                "program_name": name,
                "category":     category,
            },
        ))

    # Admission requirements (mainly master programs)
    if prog.get("admission_requirements"):
        reqs     = prog["admission_requirements"]
        req_text = f"Admission Requirements for {name}:\n\n"
        if reqs.get("bachelor_degree"):
            req_text += f"Bachelor Degree: {reqs['bachelor_degree']}\n\n"
        if reqs.get("english_proficiency"):
            ep = reqs["english_proficiency"]
            req_text += "English Proficiency Requirements:\n"
            if isinstance(ep, dict):
                if ep.get("minimum_score"):
                    req_text += f"  Minimum Score: {ep['minimum_score']}\n"
                if ep.get("accepted_tests"):
                    for t in ep["accepted_tests"]:
                        req_text += f"  - {t}\n"
            elif isinstance(ep, list):
                for item in ep:
                    req_text += f"  - {item}\n"
            req_text += "\n"
        if reqs.get("application_documents"):
            req_text += "Application Documents:\n"
            for doc in reqs["application_documents"]:
                req_text += f"- {doc}\n"
        if reqs.get("important_notes"):
            req_text += "\nImportant Notes:\n"
            for note in reqs["important_notes"]:
                req_text += f"- {note}\n"
        docs.append(Document(
            page_content=req_text,
            metadata={
                "source":       f"{source_prefix}:{pid}:admission",
                "type":         "admission_requirements",
                "program_name": name,
                "category":     category,
            },
        ))

    # Internship requirement (standalone doc for direct retrieval)
    if prog.get("internship_requirement"):
        intern_text = (
            f"{name} - Internship Requirement:\n"
            f"{prog['internship_requirement']}"
        )
        docs.append(Document(
            page_content=intern_text,
            metadata={
                "source":       f"{source_prefix}:{pid}:internship",
                "type":         "internship",
                "program_name": name,
                "category":     category,
            },
        ))

    # Last-semester optional tracks (standalone doc)
    if prog.get("last_semester_optional_tracks"):
        tracks_text = (
            f"{name} - Final Semester Track Options:\n"
            + "\n".join(f"- {t}" for t in prog["last_semester_optional_tracks"])
        )
        docs.append(Document(
            page_content=tracks_text,
            metadata={
                "source":       f"{source_prefix}:{pid}:tracks",
                "type":         "tracks",
                "program_name": name,
                "category":     category,
            },
        ))

    return docs


# ── Comprehensive JSON loader ─────────────────────────────────────────────────

def _load_comprehensive_docs() -> list:
    if not COMPREHENSIVE_PATH.exists():
        print(f"Warning: {COMPREHENSIVE_PATH} not found, skipping structured program docs.")
        return []

    with open(COMPREHENSIVE_PATH, encoding="utf-8") as f:
        data = json.load(f)

    docs = []

    # Institution-level summary
    inst_text = (
        f"Institution: {data['institution']}\n"
        f"Total Programs: {data['total_programs']}\n"
        f"  Honor Bachelor Programs: {data['programs_summary']['honor_bachelor_count']}\n"
        f"  Bachelor Programs: {data['programs_summary']['bachelor_count']}\n"
        f"  Master Programs: {data['programs_summary']['master_count']}\n"
    )
    # Also list all programs explicitly for retrieval
    inst_text += "\nAcademic Programs offered at Faculty of Engineering RUPP:\n"
    for prog in data.get("honor_bachelor_programs", []):
        inst_text += f"  (Honor Bachelor) {prog['name']}\n"
    for prog in data.get("bachelor_programs", []):
        inst_text += f"  (Bachelor) {prog['name']}\n"
    for prog in data.get("master_programs", []):
        inst_text += f"  (Master) {prog['name']}\n"

    docs.append(Document(
        page_content=inst_text,
        metadata={"source": "institution", "type": "institution_overview"},
    ))

    # Grading scale
    gs = data.get("grading_scale", {})
    if gs:
        gs_text = f"FE-RUPP Grading Scale ({gs.get('scale', '4.00 GPA')}):\n"
        for g in gs.get("grades", []):
            gs_text += f"  {g['score_range']}% → {g['grade']} (GPA {g['gpa']}) - {g['description']}\n"
        if gs.get("note"):
            gs_text += f"\nNote: {gs['note']}"
        docs.append(Document(
            page_content=gs_text,
            metadata={"source": "institution", "type": "grading_scale"},
        ))

    # General info
    gi = data.get("general_info", {})
    if gi:
        gi_text = "FE-RUPP General Information:\n\n"
        if gi.get("admission_requirements"):
            gi_text += f"Admission Requirements: {gi['admission_requirements']}\n\n"
        if gi.get("scholarship_info"):
            gi_text += f"Scholarships: {gi['scholarship_info']}\n\n"
        if gi.get("internship_requirement"):
            gi_text += f"Internship: {gi['internship_requirement']}\n"
        if gi.get("contact"):
            c = gi["contact"]
            gi_text += (
                f"\nContact:\n  Address: {c.get('address', '')}\n"
                f"  Phone: {c.get('phone', '')}\n  Email: {c.get('email', '')}\n"
            )
        docs.append(Document(
            page_content=gi_text,
            metadata={"source": "general_info", "type": "general"},
        ))

    # Location and contact (optimised for "where is", "address", "how to contact")
    about   = data.get("about_us", {})
    contact = about.get("contact", {})
    if contact:
        location = contact.get("location", "")
        loc_text = (
            "FE-RUPP Location and Contact Information:\n\n"
            f"Where is FE-RUPP located?\n"
            f"The Faculty of Engineering (FE) at RUPP is located at:\n"
            f"  Office: {contact.get('office', '')}\n"
            f"  Location: {location}\n"
            f"  City: Phnom Penh, Cambodia\n\n"
            f"Full address: {location}, Phnom Penh, Cambodia\n\n"
            f"FE-RUPP is situated in Phnom Penh, Cambodia — "
            f"specifically in Room 103, STEM Building, Royal University of Phnom Penh (Campus 1), "
            f"Russian Federation Blvd (110), Phnom Penh.\n\n"
            f"How to contact FE-RUPP:\n"
            f"  Email: {contact.get('email', '')}\n"
            f"  Phone: {contact.get('phone', '')}\n"
        )
        docs.append(Document(
            page_content=loc_text,
            metadata={"source": "about_us:contact", "type": "location_contact"},
        ))

    # Teaching framework / CDIO / about FE
    dean_msg = about.get("dean_message", {})
    if dean_msg:
        cdio_text = (
            "FE-RUPP Teaching Framework and About:\n\n"
            f"Dean: {dean_msg.get('dean_name', '')}, {dean_msg.get('title', '')}\n"
            f"Established: {dean_msg.get('established', '')}\n\n"
            "Teaching Methodology and Framework:\n"
            "FE-RUPP uses the CDIO (Conceive, Design, Implement, Operate) framework "
            "to enhance engineering education and build programs toward international level.\n"
            f"Teaching framework: {dean_msg.get('teaching_framework', '')}\n\n"
            f"About the Faculty:\n{dean_msg.get('message', '')}\n"
        )
        docs.append(Document(
            page_content=cdio_text,
            metadata={"source": "about_us:dean_message", "type": "about_teaching"},
        ))

    # Soft skills (separate doc for direct retrieval)
    soft_skills = about.get("soft_skills", {})
    if soft_skills:
        ss_text = (
            "FE-RUPP Soft Skills Program:\n\n"
            f"Skills taught: {', '.join(soft_skills.get('skills', []))}\n"
            f"Hours per week: {soft_skills.get('hours_per_week', '')}\n"
            f"Duration: {soft_skills.get('duration', '')}\n"
            f"Certification: {soft_skills.get('certification', '')}\n"
        )
        docs.append(Document(
            page_content=ss_text,
            metadata={"source": "about_us:soft_skills", "type": "soft_skills"},
        ))

    # Per-program documents
    for prog in data.get("honor_bachelor_programs", []):
        docs.extend(_docs_for_program(prog, "honor_bachelor"))

    for prog in data.get("bachelor_programs", []):
        docs.extend(_docs_for_program(prog, "bachelor"))

    for prog in data.get("master_programs", []):
        docs.extend(_docs_for_program(prog, "master"))

    return docs


# ── Index builder ─────────────────────────────────────────────────────────────

def build_chroma_index() -> None:
    print(f"Loading chunks from {CHUNKS_PATH} …")
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_PATH}")

    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks_data = json.load(f)

    if not chunks_data:
        raise ValueError("Chunks file is empty — run the scraper first.")

    docs = [
        Document(
            page_content=item["text"],
            metadata=_clean_metadata(item.get("metadata", {})),
        )
        for item in chunks_data
        if item.get("text", "").strip()
    ]
    print(f"Loaded {len(docs)} chunks from scraper.")

    program_docs = _load_comprehensive_docs()
    docs.extend(program_docs)
    print(f"Added {len(program_docs)} structured program documents.")
    print(f"Total: {len(docs)} documents.")

    if DB_DIR.exists():
        print(f"Removing old ChromaDB at {DB_DIR} …")
        shutil.rmtree(DB_DIR)

    print(f"Initialising embedding model ({EMBEDDING_MODEL}) …")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"Building ChromaDB index at {DB_DIR} …")
    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
    )
    print(f"Done — {len(docs)} total documents stored in ChromaDB.")


if __name__ == "__main__":
    build_chroma_index()
