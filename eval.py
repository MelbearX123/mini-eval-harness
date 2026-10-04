"""Mini eval harness: vet visit transcript -> SOAP note (JSON) -> code-based checks.

Usage:
    pip install anthropic python-dotenv   # put ANTHROPIC_API_KEY=... in .env
    python eval.py run v1          # run every case with prompt v1, save results/v1.json
    python eval.py run v2          # same with prompt v2
    python eval.py compare v1 v2   # which cases got fixed / broke between the two runs
"""

import argparse
import json
import re
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()  # picks up ANTHROPIC_API_KEY from .env

MODEL = "claude-haiku-4-5"
RESULTS_DIR = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Prompts: edit v2 (or add v3...) and rerun to compare.
# ---------------------------------------------------------------------------

PROMPTS = {
    "v1": (
        "You are a veterinary scribe. Write a SOAP note from the visit transcript "
        "using the write_soap_note tool."
    ),
    "v2": (
        "You are a veterinary scribe. Write a SOAP note from the visit transcript "
        "using the write_soap_note tool.\n\n"
        "Rules:\n"
        "- Only record vital signs and measurements the vet explicitly stated, with the exact "
        "numbers. If a vital wasn't measured, leave it out entirely (don't write 'not recorded').\n"
        "- The plan must name every medication prescribed, with dose and frequency.\n"
        "- If the vet changes their mind, record only the final decision.\n"
        "- The medications list contains only drugs actually prescribed or continued for this "
        "patient. Exclude drugs that were discussed but declined, cancelled, only warned about, "
        "or taken by the owner."
    ),
}

SOAP_TOOL = {
    "name": "write_soap_note",
    "description": "Record a SOAP note for a veterinary visit.",
    "input_schema": {
        "type": "object",
        "properties": {
            "subjective": {"type": "string", "description": "History and owner-reported concerns."},
            "objective": {"type": "string", "description": "Exam findings, vitals, test results."},
            "assessment": {"type": "string", "description": "Diagnosis or differential diagnoses."},
            "plan": {"type": "string", "description": "Treatments, medications, follow-up."},
            "medications": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Medications prescribed or continued for the patient.",
            },
        },
        "required": ["subjective", "objective", "assessment", "plan", "medications"],
    },
}

REQUIRED_FIELDS = ["subjective", "objective", "assessment", "plan", "medications"]

# Generic + brand names, so "Rimadyl" counts as carprofen.
MED_ALIASES = {
    "carprofen": ["carprofen", "rimadyl", "novox", "vetprofen"],
    "gabapentin": ["gabapentin", "neurontin"],
    "otomax": ["otomax", "gentamicin", "betamethasone", "clotrimazole"],
    "fish oil": ["fish oil", "omega-3", "omega 3"],
    "heartgard": ["heartgard", "ivermectin"],
    "laxatone": ["laxatone", "hairball"],
    "ibuprofen": ["ibuprofen", "advil", "motrin"],
    "cephalexin": ["cephalexin", "keflex"],
    "cefpodoxime": ["cefpodoxime", "simplicef"],
    "insulin glargine": ["glargine", "lantus", "insulin"],
    "chlorhexidine": ["chlorhexidine"],
    "apomorphine": ["apomorphine"],
    "activated charcoal": ["activated charcoal", "charcoal"],
    "mirtazapine": ["mirtazapine", "mirataz"],
    "subcutaneous fluids": ["subcutaneous fluid", "subq fluid", "sub-q fluid", "sq fluid", "lactated ringer"],
    "maropitant": ["maropitant", "cerenia"],
    "oclacitinib": ["oclacitinib", "apoquel"],
}

# What counts as "mentioning" a vital in the objective section.
VITAL_PATTERNS = {
    "temperature": r"\btemp(erature)?\b|°\s*f\b|\bfebrile\b|\bafebrile\b",
    "heart_rate": r"\bheart rate\b|\bhr\b|\bbpm\b|\bpulse\b",
    "respiratory_rate": r"\brespiratory rate\b|\bresp(iration)?s?\b|\brr\b|\bbreaths\b",
    "weight": r"\bweigh(t|s|ed)?\b|\bkg\b|\blbs?\b",
}

# ---------------------------------------------------------------------------
# Test cases. You wrote the transcripts, so you know the ground truth:
#   meds      - must appear in the plan and the medications list
#   not_meds  - mentioned in the transcript but must NOT be in the medications list
#   vitals    - the only vitals the vet actually stated
# ---------------------------------------------------------------------------

CASES = [
    {
        "id": "routine_checkup",
        "transcript": """\
Vet: Hi, this is Bella? Six-year-old Lab, here for her annual.
Owner: Yep, she's doing great, no complaints.
Vet: Good. She weighs 28.4 kg today, about the same as last year. Temperature is 101.5, heart rate 92.
Vet: Eyes, ears, teeth look good. Mild tartar on the upper molars. Heart and lungs sound clear. Abdomen soft.
Vet: She's due for rabies and DHPP, we'll do those today. Is she still on Heartgard?
Owner: Yes, once a month.
Vet: Great, keep giving the Heartgard monthly. I'll send home a refill. See you next year.""",
        "meds": ["heartgard"],
        "not_meds": [],
        "vitals": ["temperature", "heart_rate", "weight"],
    },
    {
        "id": "many_medications",
        "transcript": """\
Vet: Max is 12 now, right? What's going on?
Owner: He's really stiff getting up in the morning, and he keeps shaking his head.
Vet: Let's look. He's 34 kg. His hips are painful on extension, especially the left. Right ear canal is red with brown discharge, left ear looks fine.
Vet: Cytology of the right ear shows yeast and some cocci.
Vet: For the arthritis, start carprofen 75 mg twice daily with food. Add gabapentin 100 mg at night for pain.
Vet: For the ear, Otomax, 4 drops in the right ear twice daily for 7 days.
Vet: Keep giving the fish oil he's on, that helps the joints.
Vet: Bloodwork in 2 weeks to check his liver on the carprofen. Recheck the ear then too.""",
        "meds": ["carprofen", "gabapentin", "otomax", "fish oil"],
        "not_meds": [],
        "vitals": ["weight"],
    },
    {
        "id": "rambling_owner",
        "transcript": """\
Vet: So what brings Luna in today?
Owner: Oh gosh, well, we just got back from my sister's place in Tucson, which, by the way, was so hot, I don't know how people live there. Anyway Luna stayed with my neighbor, and she's lovely but she feeds her the cheap food, and since we got back Luna's been throwing up. Hairballs mostly I think. Maybe 3 times this week? My back's been killing me from the drive so I've been popping ibuprofen like candy, ha.
Vet: Okay. Any changes in appetite or litter box?
Owner: No, she eats like a horse. Did I tell you my sister got a parrot? It talks. Mostly swears.
Vet: Ha. Luna weighs 4.3 kg. Belly is soft, no pain, she's well hydrated. Coat is a bit matted on her back.
Vet: This looks like hairballs. Let's try Laxatone, about an inch of gel once a day for a week, then twice a week. Brush her daily.
Vet: And please keep your ibuprofen away from her. It's toxic to cats, even a single tablet.
Owner: Oh my god, I had no idea.
Vet: If the vomiting continues past 2 weeks or she stops eating, bring her back.""",
        "meds": ["laxatone"],
        "not_meds": ["ibuprofen"],
        "vitals": ["weight"],
    },
    {
        "id": "vet_changes_mind",
        "transcript": """\
Vet: Cooper has these red bumps and crusts all over his belly. That looks like a superficial pyoderma.
Vet: I'm going to start him on cephalexin, 500 mg twice daily for 3 weeks.
Owner: Wait, is that the one he had last year? He had terrible diarrhea on it.
Vet: Let me check his record. Yes, it was cephalexin. Okay, let's not use that.
Vet: Instead we'll do cefpodoxime, Simplicef, 200 mg once a day for 3 weeks. Once a day is easier anyway.
Vet: Bathe him with the chlorhexidine shampoo twice a week.
Vet: He's 23 kg. Recheck in 3 weeks.""",
        "meds": ["cefpodoxime", "chlorhexidine"],
        "not_meds": ["cephalexin"],
        "vitals": ["weight"],
    },
    {
        "id": "no_vitals_limping",
        "transcript": """\
Owner: Rex started limping on his back right leg yesterday after chasing a squirrel.
Vet: Let me feel it. He's sore when I flex the right knee. No drawer sign, the ligament feels stable. No swelling.
Vet: Probably a soft tissue strain. Let's start him on Rimadyl, 75 mg twice daily for 5 days.
Vet: Strict leash walks only, no running for 2 weeks. If he's not better in a week, we'll take X-rays.""",
        "meds": ["carprofen"],
        "not_meds": [],
        "vitals": [],
    },
    {
        "id": "diabetic_cat_dose_change",
        "transcript": """\
Vet: How's Mittens doing on the insulin?
Owner: She's still drinking a ton and peeing a lot.
Vet: Her weight is 5.1 kg, down from 5.4. Heart rate 190. Her glucose curve today peaked at 410 and the lowest was 280. That's too high.
Vet: Let's increase her glargine from 2 units to 3 units twice daily.
Vet: Keep the same diet. Come back in 1 week for another glucose curve.""",
        "meds": ["insulin glargine"],
        "not_meds": [],
        "vitals": ["weight", "heart_rate"],
    },
    {
        "id": "dental_planning",
        "transcript": """\
Vet: Rocky's breath is pretty bad, and there's heavy tartar and red gums all along the back teeth.
Vet: Temperature is 101.8. Lower left premolar looks loose.
Vet: He needs a dental cleaning under anesthesia, probably with an extraction. We'll do pre-anesthetic bloodwork today.
Vet: Until then, use the chlorhexidine oral rinse once daily.
Owner: Does he need antibiotics?
Vet: Not right now. We'll decide during the procedure if needed.""",
        "meds": ["chlorhexidine"],
        "not_meds": [],
        "vitals": ["temperature"],
    },
    {
        "id": "chocolate_toxicity",
        "transcript": """\
Owner: Daisy ate most of a dark chocolate bar about 45 minutes ago!
Vet: Okay, let's move fast. She's 9 kg. Heart rate 160, temperature 102.4, respiratory rate 40. She's restless and panting.
Vet: That's a toxic dose for her size. I'm giving apomorphine now to make her vomit.
Vet: She vomited, lots of chocolate came up. Now we'll give activated charcoal, 1 g per kg, by mouth.
Vet: We'll keep her on IV fluids and monitor her heart for the next 6 hours.""",
        "meds": ["apomorphine", "activated charcoal"],
        "not_meds": [],
        "vitals": ["heart_rate", "temperature", "respiratory_rate", "weight"],
    },
    {
        "id": "ckd_cat_held_med",
        "transcript": """\
Vet: Oliver's kidney values are worse. Creatinine is 4.2 now, it was 3.1 in March.
Owner: He's barely eating.
Vet: He's lost weight, he's 3.6 kg. He feels a little dehydrated.
Vet: Let's start subcutaneous fluids at home, 100 mL every other day. We'll show you how.
Vet: And mirtazapine ointment, Mirataz, a 1.5 inch strip on his ear once a day to help his appetite.
Vet: I thought about adding Cerenia for nausea, but let's hold off and see how the mirtazapine does first.
Vet: Recheck bloodwork in 1 month.""",
        "meds": ["subcutaneous fluids", "mirtazapine"],
        "not_meds": ["maropitant"],
        "vitals": ["weight"],
    },
    {
        "id": "brand_name_only",
        "transcript": """\
Owner: Bailey won't stop scratching and licking her paws.
Vet: Her paws are red between the toes and she has some hair loss on her belly. No fleas seen. Looks like environmental allergies.
Vet: Let's start Apoquel, 16 mg twice daily for 14 days, then once a day.
Vet: Wipe her paws after walks. Recheck in 1 month.""",
        "meds": ["oclacitinib"],
        "not_meds": [],
        "vitals": [],
    },
]

# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment


def generate_note(transcript: str, system_prompt: str) -> dict:
    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=system_prompt,
        tools=[SOAP_TOOL],
        tool_choice={"type": "tool", "name": "write_soap_note"},
        messages=[{"role": "user", "content": transcript}],
        # SDK 1.x dropped the temperature kwarg, but Haiku 4.5 still honors it.
        # temperature=0 reduces run-to-run noise so v1 vs v2 differences mean something.
        extra_body={"temperature": 0},
    )
    if response.stop_reason == "max_tokens":
        raise RuntimeError("hit max_tokens; tool input is truncated")
    for block in response.content:
        if block.type == "tool_use":
            return block.input
    raise RuntimeError("model did not call write_soap_note")


# ---------------------------------------------------------------------------
# Checks: each returns (passed, detail)
# ---------------------------------------------------------------------------

def mentions(text: str, med: str) -> bool:
    text = text.lower()
    return any(alias in text for alias in MED_ALIASES.get(med, [med]))


def check_required_fields(note, case):
    missing = [f for f in REQUIRED_FIELDS if f not in note]
    empty = [f for f in REQUIRED_FIELDS if f in note and f != "medications" and not str(note[f]).strip()]
    ok = not missing and not empty
    return ok, "" if ok else f"missing={missing} empty={empty}"


def check_meds_in_plan(note, case):
    missing = [m for m in case["meds"] if not mentions(note.get("plan", ""), m)]
    return not missing, f"not in plan: {missing}" if missing else ""


def check_meds_list_complete(note, case):
    meds_text = " | ".join(note.get("medications", []))
    missing = [m for m in case["meds"] if not mentions(meds_text, m)]
    return not missing, f"not in medications list: {missing}" if missing else ""


def check_no_unprescribed_meds(note, case):
    meds_text = " | ".join(note.get("medications", []))
    present = [m for m in case["not_meds"] if mentions(meds_text, m)]
    return not present, f"should not be listed: {present}" if present else ""


def check_no_invented_vitals(note, case):
    objective = note.get("objective", "").lower()
    invented = [
        vital for vital, pattern in VITAL_PATTERNS.items()
        if vital not in case["vitals"] and re.search(pattern, objective)
    ]
    return not invented, f"vitals not in transcript: {invented}" if invented else ""


def check_no_invented_numbers(note, case):
    # Any number in the objective section must appear somewhere in the transcript.
    number = r"\d+(?:\.\d+)?"
    allowed = set(re.findall(number, case["transcript"]))
    invented = sorted(set(re.findall(number, note.get("objective", ""))) - allowed)
    return not invented, f"numbers not in transcript: {invented}" if invented else ""


CHECKS = {
    "required_fields": check_required_fields,
    "meds_in_plan": check_meds_in_plan,
    "meds_list_complete": check_meds_list_complete,
    "no_unprescribed_meds": check_no_unprescribed_meds,
    "no_invented_vitals": check_no_invented_vitals,
    "no_invented_numbers": check_no_invented_numbers,
}

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(version: str) -> None:
    system_prompt = PROMPTS[version]
    results = []
    for case in CASES:
        try:
            note = generate_note(case["transcript"], system_prompt)
            checks = {name: fn(note, case) for name, fn in CHECKS.items()}
        except (anthropic.APIError, RuntimeError) as e:
            note = None
            checks = {"api_call": (False, str(e))}
        passed = all(ok for ok, _ in checks.values())
        results.append({
            "id": case["id"],
            "passed": passed,
            "checks": {name: {"ok": ok, "detail": detail} for name, (ok, detail) in checks.items()},
            "note": note,
        })
        print(f"{'PASS' if passed else 'FAIL'}  {case['id']}")
        for name, (ok, detail) in checks.items():
            if not ok:
                print(f"        {name}: {detail}")

    n_passed = sum(r["passed"] for r in results)
    print(f"\n[{version}] {n_passed}/{len(results)} cases passed")
    print("Per-check pass rate:")
    for name in CHECKS:
        n = sum(r["checks"].get(name, {}).get("ok", False) for r in results)
        print(f"  {name:<22} {n}/{len(results)}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{version}.json"
    out.write_text(json.dumps({"version": version, "model": MODEL, "prompt": system_prompt,
                               "results": results}, indent=2))
    print(f"Saved {out}")


def compare(old: str, new: str) -> None:
    load = lambda v: {r["id"]: r for r in json.loads((RESULTS_DIR / f"{v}.json").read_text())["results"]}
    a, b = load(old), load(new)
    print(f"{old}: {sum(r['passed'] for r in a.values())}/{len(a)}   "
          f"{new}: {sum(r['passed'] for r in b.values())}/{len(b)}\n")
    for case_id in a:
        if case_id not in b:
            continue
        before, after = a[case_id]["passed"], b[case_id]["passed"]
        label = {(False, True): "FIXED", (True, False): "BROKE"}.get((before, after),
                                                                    "pass" if after else "fail")
        print(f"{label:<6} {case_id}")
        # Show check-level changes, which matter even when the case stays failing.
        for name in CHECKS:
            ca, cb = a[case_id]["checks"].get(name), b[case_id]["checks"].get(name)
            if ca and cb and ca["ok"] != cb["ok"]:
                print(f"         {name}: {'fail -> pass' if cb['ok'] else 'pass -> fail'} {cb['detail']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("version", choices=PROMPTS.keys())
    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("old")
    p_cmp.add_argument("new")
    args = parser.parse_args()

    if args.cmd == "run":
        run(args.version)
    else:
        compare(args.old, args.new)


if __name__ == "__main__":
    sys.exit(main())
