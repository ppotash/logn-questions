"""Prompt construction and response parsing for the log(N)-Questions game.

This module is the single source of truth for what every model sees. Providers
differ only in transport; if wording ever varies per provider, the comparison is
confounded. Nothing here imports a provider SDK.

Layout is deliberate. Each prompt is returned as three parts:

    system      role and rules; identical for every call of that role
    cacheable   the document block; identical across every round of a game
    tail        round number and question history; the only part that changes

Providers place their cache breakpoint at the end of `cacheable`. Keeping the
documents ahead of the history is what makes the prefix stable.

The two role prompts are written to agree with each other. Questioner and
answerer agreement compounds as p^log2(N), so any gap between what the
questioner assumes the answerer will do and what the answerer is actually told
is a direct tax on the result. The questioner is therefore told the answerer's
rules explicitly.

PROMPT_VERSION is hashed from the templates below. Record it in every result
file: if it changes, results from before and after are not comparable.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


# --------------------------------------------------------------------------
# templates
# --------------------------------------------------------------------------

# The "cannot answer well" list is empirical. Observed failures at N=32:
#   "Is the subject primarily associated with Spain rather than Italy?"
# answered No about a document opening "Vocento, S.A. ... is a Spanish mass
# media group". Italy was in the question only because another candidate was
# Italian; the answerer, holding one document, could not tell why. The
# questioner had bisected perfectly and still lost the game.
QBOT_SYSTEM = """\
You are playing a deduction game.

There is a numbered set of documents. One of them has been secretly chosen as \
the target. Your job is to identify it.

You may ask exactly {rounds} yes/no questions, one per round. After the final \
answer you will be asked to name the target.

Your questions are sent to an answerer who can see ONLY the target document \
and your question. It does not know which other documents exist, how many \
there are, or the target's number.

The answerer is instructed to judge your question against the document's \
subject using ordinary world knowledge, so it will answer Yes to "is it \
European?" for a document about a Spanish company even though the document \
never says "Europe". You can rely on that.

Two kinds of question it cannot answer well:

  - anything referring to document numbers, positions, or the set as a whole
  - anything that contrasts the subject with an alternative drawn from the \
other documents. This covers every phrasing of the contrast -- "X rather than \
Y", "X as opposed to Y", "X instead of Y", "X, not Y", "X versus Y" -- and \
also parenthetical asides of the same shape. The alternative exists only in \
your head: the answerer holds one document, cannot see why you raised Y, and \
may simply answer No.

So ask about one property, in one clause, with nothing to contrast it \
against. Instead of "is it a concept as opposed to a person", ask "is the \
subject an abstract concept". Instead of "is it Spanish rather than Italian", \
ask "is the subject Spanish". A single property can still group many \
documents at once, which is what you want.

The answerer always replies Yes or No, and never anything else.

Strategy: each answer should eliminate as many documents as possible, so aim \
for questions that divide the documents still consistent with the answers so \
far into two groups of roughly equal size. A question the answerer will judge \
the way you intend is worth more than a perfectly balanced one it might get \
wrong. You are not given a running list of which documents remain in \
contention; work that out yourself each round from the full document set and \
the answers so far.

Respond with your reasoning if you wish, then end your reply with a line in \
exactly this form:

QUESTION: <your yes/no question>

The QUESTION line must be the last line of your reply and must contain a \
single question."""

QBOT_GUESS_SYSTEM = """\
You are playing a deduction game.

There is a numbered set of documents. One of them was secretly chosen as the \
target. You have used all {rounds} of your yes/no questions and must now name \
the target.

The answers you received are truthful. Work out which document is consistent \
with all of them.

Respond with your reasoning if you wish, then end your reply with a line in \
exactly this form:

GUESS: <document number>

The GUESS line must be the last line of your reply and must contain a single \
number between 1 and {n}."""

# Three things this prompt has to get right, all learned from failures.
#
# World knowledge: an early version said "answer on the basis of the document
# alone", which reads as a prohibition on inference. A Spanish media group's
# article does not contain "Spain is in Europe", so a literal reading answered
# No to a Europe question and silently eliminated the target.
#
# Comparatives: "X rather than Y" where Y is absent from the document was
# answered No. Naming only that one surface form was not enough -- the next
# failure was "related to leadership (as opposed to equipment or machinery)"
# answered No about a document titled "Task-oriented and relationship-oriented
# leadership". The rule has to cover every phrasing of a contrast, and the
# instruction is to ignore the contrast rather than to adjudicate it.
#
# No-bias: with a one-word answer forced, models default to No when uncertain.
# Measured 21% Yes across 24 answers where ~50% was expected (p = 0.003).
ABOT_SYSTEM = """\
You will be shown one document and one yes/no question about its subject.

Answer with exactly one word: Yes or No.

Use ordinary world knowledge. The document will not always state the answer \
outright: if it describes a Spanish company, a question about Europe is Yes; \
if it describes a golfer, a question about sport is Yes. Do not require the \
document to contain the question's wording.

Questions may be vague, compound, or only loosely applicable. Do your best \
with them rather than refusing:

  - if a question contrasts the subject with an alternative -- "X rather than \
Y", "X as opposed to Y", "X instead of Y", "X, not Y", or the same contrast in \
a parenthetical -- ignore the contrast entirely and judge only whether the \
subject fits X. Y is not in this document and is not supposed to be; its \
absence is never a reason to answer No. A document titled "Task-oriented and \
relationship-oriented leadership" answers Yes to "is this about leadership, as \
opposed to machinery".
  - if a question is metaphorical or only partly applicable, judge whether it \
is a fair description of this subject overall.

Whatever the question, one of Yes or No describes this document better than \
the other. Choose it. Yes and No are equally acceptable answers: do not fall \
back on No when uncertain.

Never reply with anything other than Yes or No, and never explain."""

DOCS_HEADER = "The documents:\n"

ROUND_TAIL = """\
Round {round} of {rounds}.

{history}
Ask your question for this round."""

GUESS_TAIL = """\
All {rounds} rounds are complete.

{history}
Name the target document."""

NO_HISTORY = "No questions have been asked yet.\n"

ABOT_TAIL = """\
Document:
{document}

Question: {question}"""

# Sent after a reply that could not be parsed. Kept minimal so it does not
# teach anything about the game beyond the format requirement.
REPAIR_QUESTION = ("Your reply did not end with a QUESTION: line. "
                   "Reply again, ending with QUESTION: followed by your question.")
REPAIR_GUESS = ("Your reply did not end with a GUESS: line. "
                "Reply again, ending with GUESS: followed by a single document number.")
REPAIR_ANSWER = "Reply with exactly one word: Yes or No."


PROMPT_VERSION = hashlib.sha256(
    "\x00".join([
        QBOT_SYSTEM, QBOT_GUESS_SYSTEM, ABOT_SYSTEM, DOCS_HEADER,
        ROUND_TAIL, GUESS_TAIL, NO_HISTORY, ABOT_TAIL,
    ]).encode("utf-8")
).hexdigest()[:12]


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Prompt:
    """A prompt split at the cache boundary.

    `cacheable` is byte-identical across every round of a game, so providers
    should mark a cache breakpoint at its end. `tail` changes each round.
    """
    system: str
    cacheable: str
    tail: str

    def as_text(self) -> str:
        """Flat rendering, for providers without a block-based cache API."""
        return f"{self.cacheable}\n\n{self.tail}"


def render_documents(docs: list[dict]) -> str:
    """Number documents from 1. Title and text are joined so that title-based
    questions are decidable by the answerer, which sees the same rendering."""
    lines = [DOCS_HEADER]
    for i, d in enumerate(docs, 1):
        lines.append(f"[{i}] {d['title']}\n{d['text']}\n")
    return "\n".join(lines)


def render_history(history: list[tuple[str, bool]]) -> str:
    """history: [(question, answer_is_yes), ...] in round order."""
    if not history:
        return NO_HISTORY
    out = ["Questions asked so far:"]
    for i, (q, a) in enumerate(history, 1):
        out.append(f"  {i}. {q}\n     Answer: {'Yes' if a else 'No'}")
    return "\n".join(out) + "\n"


def qbot_prompt(docs: list[dict], history: list[tuple[str, bool]],
                round_idx: int, rounds: int) -> Prompt:
    """round_idx is 1-based."""
    return Prompt(
        system=QBOT_SYSTEM.format(rounds=rounds),
        cacheable=render_documents(docs),
        tail=ROUND_TAIL.format(round=round_idx, rounds=rounds,
                               history=render_history(history)),
    )


def qbot_guess_prompt(docs: list[dict], history: list[tuple[str, bool]],
                      rounds: int) -> Prompt:
    return Prompt(
        system=QBOT_GUESS_SYSTEM.format(rounds=rounds, n=len(docs)),
        cacheable=render_documents(docs),
        tail=GUESS_TAIL.format(rounds=rounds, history=render_history(history)),
    )


def abot_prompt(doc: dict, question: str) -> Prompt:
    """The answerer sees the target rendered exactly as the questioner sees it,
    minus the number. Withholding the number is what stops the game collapsing
    into integer bisection; keeping the title is what leaves the enumeration and
    lexical strategies genuinely available."""
    return Prompt(
        system=ABOT_SYSTEM,
        cacheable="",
        tail=ABOT_TAIL.format(document=f"{doc['title']}\n{doc['text']}",
                              question=question),
    )


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Parsed:
    """Parse outcome. `mode` records how the value was recovered:

        "strict"   the model followed the requested format
        "loose"    recovered from an unformatted reply
        "failed"   nothing usable

    Log the mode. Format compliance is a real difference between models, and
    silently rescuing malformed replies would hide it.
    """
    value: object
    mode: str

    @property
    def ok(self) -> bool:
        return self.mode != "failed"


_QUESTION_TAG = re.compile(r"^\s*QUESTION\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_GUESS_TAG = re.compile(r"^\s*GUESS\s*:\s*\[?\s*(\d+)", re.IGNORECASE | re.MULTILINE)
_YESNO = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def parse_question(text: str) -> Parsed:
    if not text:
        return Parsed(None, "failed")
    matches = _QUESTION_TAG.findall(text)
    if matches:
        q = matches[-1].strip().strip('"')
        if q:
            return Parsed(q, "strict")
    # Loose: the last line that reads like a question.
    for line in reversed([ln.strip() for ln in text.splitlines() if ln.strip()]):
        if line.endswith("?"):
            return Parsed(line.lstrip("*-# ").strip('"'), "loose")
    return Parsed(None, "failed")


def parse_guess(text: str, n: int) -> Parsed:
    if not text:
        return Parsed(None, "failed")
    matches = _GUESS_TAG.findall(text)
    if matches:
        v = int(matches[-1])
        if 1 <= v <= n:
            return Parsed(v, "strict")
    # Loose: the last in-range integer anywhere in the reply.
    for tok in reversed(re.findall(r"\d+", text)):
        v = int(tok)
        if 1 <= v <= n:
            return Parsed(v, "loose")
    return Parsed(None, "failed")


def parse_answer(text: str) -> Parsed:
    """True for yes, False for no."""
    if not text:
        return Parsed(None, "failed")
    stripped = text.strip().strip(".!,*\"' ").lower()
    if stripped in ("yes", "no"):
        return Parsed(stripped == "yes", "strict")
    # Loose mode takes the LAST match, not the first. Observed in the wild:
    #   "No... wait. The document concerns biathlon, a winter sport. Yes"
    # A model that talks itself out of an answer has corrected itself, and the
    # correction is what it meant. Taking the first match recorded the exact
    # opposite of the model's conclusion.
    matches = list(_YESNO.finditer(text))
    if matches:
        return Parsed(matches[-1].group(1).lower() == "yes", "loose")
    return Parsed(None, "failed")


# --------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------

if __name__ == "__main__":
    docs = [{"title": f"Doc {i}", "text": f"Body of document {i}."} for i in range(1, 5)]
    hist = [("Is the subject a person?", False), ("Was it built before 1900?", True)]

    print("=" * 70, "\nQ-BOT SYSTEM\n", "=" * 70, sep="")
    print(qbot_prompt(docs, hist, 3, 4).system)
    print("\n" + "=" * 70, "\nA-BOT SYSTEM\n", "=" * 70, sep="")
    print(abot_prompt(docs[0], "q?").system)

    print("\n" + "=" * 70, "\nPARSING\n", "=" * 70, sep="")
    cases = [
        ("reasoning...\nQUESTION: Is it about music?", parse_question),
        ("QUESTION: A?\nQUESTION: B?", parse_question),
        ("I think:\nIs it about music?", parse_question),
        ("no idea", parse_question),
        ("thinking\nGUESS: 3", lambda t: parse_guess(t, 4)),
        ("it must be 2", lambda t: parse_guess(t, 4)),
        ("GUESS: 99", lambda t: parse_guess(t, 4)),
        ("Yes", parse_answer),
        ("no.", parse_answer),
        ("Based on the document, yes.", parse_answer),
        ("No... wait. The document concerns biathlon, a winter sport. Yes",
         parse_answer),
        ("unclear", parse_answer),
    ]
    for text, fn in cases:
        r = fn(text)
        print(f"  {r.mode:<7} {r.value!r:<28} <- {text[:48]!r}")

    print(f"\nPROMPT_VERSION = {PROMPT_VERSION}")