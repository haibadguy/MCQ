"""
Leaf – Quick demo / smoke test script
======================================
Runs MCQ generation for BOTH English and Vietnamese contexts
without starting the Flask server.

Usage:
    python main.py
"""

import textwrap
from app.mcq_generation import MCQGenerator
from app.models.question import Question


# ── Helpers ──────────────────────────────────────────────────────────────────

PIPELINE_FLAGS = {'en': '🇬🇧 English T5', 'vi': '🇻🇳 ViT5 Vietnamese'}


def show_result(question: Question):
    """Pretty-print a single MCQ."""
    flag = PIPELINE_FLAGS.get(question.lang, question.lang.upper())
    print(f"  [{flag}]")
    print(f"  Q: {question.questionText}")
    print(f"  A: {question.answerText}")
    if question.distractors:
        print(f"  D: {' | '.join(question.distractors)}")
    print()


def run_demo(generator: MCQGenerator, title: str, context: str, count: int):
    """Generate MCQs for a given context and print results."""
    print("=" * 70)
    print(f"  DEMO: {title}")
    print("=" * 70)
    print("  Context (truncated):")
    for line in textwrap.wrap(context[:300], width=65):
        print(f"    {line}")
    print(f"  Requesting {count} MCQ(s)…\n")

    questions = generator.generate_mcq_questions(context, count)
    for i, q in enumerate(questions, 1):
        print(f"  ── Question {i} ─────────────────────────────────────────")
        show_result(q)

    print(f"  Total generated: {len(questions)}\n")


# ── Contexts ──────────────────────────────────────────────────────────────────

CONTEXT_EN = (
    "The koala (Phascolarctos cinereus) is an arboreal herbivorous marsupial "
    "native to Australia. It is the only extant representative of the family "
    "Phascolarctidae and its closest living relatives are the wombats, which "
    "are members of the family Vombatidae. The koala is found in coastal areas "
    "of the mainland's eastern and southern regions, inhabiting Queensland, "
    "New South Wales, Victoria, and South Australia. It is easily recognisable "
    "by its stout, tailless body and large head with round, fluffy ears and "
    "large, spoon-shaped nose."
)

CONTEXT_VI = (
    "Việt Nam là một quốc gia nằm ở bán đảo Đông Dương thuộc khu vực Đông Nam Á, "
    "giáp với Trung Quốc ở phía bắc, Lào và Campuchia ở phía tây, Biển Đông ở phía "
    "đông và phía nam. Diện tích của Việt Nam khoảng 331.212 km² và dân số vào năm "
    "2023 ước đạt hơn 98 triệu người, đứng thứ 15 trên thế giới. Hà Nội là thủ đô "
    "và Thành phố Hồ Chí Minh là thành phố đông dân nhất. Việt Nam là thành viên của "
    "Liên Hợp Quốc, ASEAN, APEC và nhiều tổ chức quốc tế khác."
)

CONTEXT_EN_OXYGEN = (
    "Oxygen is the chemical element with the symbol O and atomic number 8. "
    "It is a member of the chalcogen group in the periodic table, a highly "
    "reactive nonmetal, and an oxidizing agent that readily forms oxides with "
    "most elements as well as with other compounds. Oxygen is Earth's most "
    "abundant element, and after hydrogen and helium, it is the third-most "
    "abundant element in the universe. At standard temperature and pressure, "
    "two atoms of the element bind to form dioxygen, a colorless and odorless "
    "diatomic gas with the formula O₂. Diatomic oxygen gas currently constitutes "
    "20.95% of the Earth's atmosphere."
)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    generator = MCQGenerator(is_verbose=True)

    # 1. English – Koala
    run_demo(generator, "English T5 Pipeline (Koala)", CONTEXT_EN, count=3)

    # 2. Vietnamese – Việt Nam
    run_demo(generator, "Vietnamese ViT5 Pipeline (Việt Nam)", CONTEXT_VI, count=3)

    # 3. English – Oxygen (longer text)
    run_demo(generator, "English T5 Pipeline (Oxygen)", CONTEXT_EN_OXYGEN, count=5)
