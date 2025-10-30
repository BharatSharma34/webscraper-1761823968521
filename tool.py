import os
import textwrap
from typing import List, Dict
from duckduckgo_search import DDGS
from openai import OpenAI

def main(user_prompt, document_ids, gpt_key):
#     """
#     Main function for web scraper1
#     This function will be called by the Lambda handler.
    
    
#     Returns:
#     dict - JSON-serializable response
#     """
#     # Your tool logic here
    print("document_ids", document_ids)
    message = run_agent(user_prompt, gpt_key)
    return {
        "success": True,
        "message": message,
        "data": {}
    }

# -------------------------
# CONFIG
# -------------------------
# Set your API key before running, or set it in Colab secrets.
def get_api_key(gpt_key):
    os.environ["OPENAI_API_KEY"] = gpt_key
    return
 # <- replace or inject securely
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

MODEL_REASONING = "gpt-4.1"        # higher reasoning quality
MODEL_LIGHT = "gpt-4.1-mini"       # cheaper/quicker, used for planning and reflections

# -------------------------
# LOW-LEVEL UTILITIES
# -------------------------

def wrap(text: str, width: int = 100) -> str:
    return textwrap.fill(text, width=width)

def web_search(query: str, max_results: int = 6, site_filters: List[str] = None) -> List[Dict[str,str]]:
    """
    Perform one or more scoped searches using a privacy-preserving meta-search layer.
    Optionally bias toward certain sources (domains) by adding `site:example.com`.
    Returns list of dicts: {title, body, link}.
    """
    results_total = []
    if site_filters:
        subqueries = [f"{query} site:{dom}" for dom in site_filters]
    else:
        subqueries = [query]

    with DDGS() as ddgs:
        for subq in subqueries:
            for r in ddgs.text(subq, max_results=max_results):
                results_total.append(
                    {
                        "title": r.get("title", ""),
                        "body": r.get("body", ""),
                        "link": r.get("href", ""),
                        "source_hint": subq
                    }
                )
    return results_total

def gpt_chat(model: str, system_msg: str, user_msg: str, temperature: float = 0.3) -> str:
    """
    Small helper to talk to the OpenAI API.
    """
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
    )
    return resp.choices[0].message.content.strip()

# -------------------------
# STEP 1: PLAN GENERATION
# -------------------------

def generate_search_plan(user_question: str) -> Dict[str, object]:
    """
    Ask the model to interpret the user's question and propose:
    - clarified intent
    - sub-questions / steps
    - search keywords
    - suggested sources/domains
    - desired output style
    We also inject language about safety/privacy of our retrieval layer.
    """

    planning_prompt = f"""
You are an assistant that builds a responsible web research plan BEFORE any live lookup.

User Question:
{user_question}

Tasks:
1. Clarify what the user is *really* asking. If it's ambiguous, make a best guess.
2. Break the work into 2-4 concrete research steps. Each step should say what to look for.
3. For each step, propose search keywords that will likely surface high-signal public information.
4. Suggest a short list of domains or source types you'd like to consult
   (e.g. 'official docs from X', 'government statistics', 'reputable tech press', 'recent academic coverage').
   Use only domain roots or descriptors, e.g. 'who.int', 'europa.eu', 'nature.com', 'arxiv.org', 'wired.com'.
5. Propose an output style (e.g. "executive summary + bullet-point findings + citations").
6. Add a one-line note that we use a privacy-preserving meta-search layer instead of directly hitting a single engine,
   which reduces personal tracking and ad targeting.

Return valid JSON with keys:
intent, steps, keywords, sources, output_style, safety_note
"""

    raw_plan = gpt_chat(
        model=MODEL_LIGHT,
        system_msg="You draft structured research plans in strict JSON.",
        user_msg=planning_prompt,
        temperature=0.2
    )

    # The model returns JSON as text. We'll try to eval it safely.
    # We'll fall back to a naive plan if parsing fails.
    import json
    try:
        plan = json.loads(raw_plan)
    except json.JSONDecodeError:
        # fallback heuristic
        plan = {
            "intent": f"Investigate: {user_question}",
            "steps": [
                "Collect recent factual background.",
                "Collect latest developments and credible expert commentary.",
                "Synthesize and cite."
            ],
            "keywords": [user_question],
            "sources": ["wikipedia.org", "reputable news outlets", "relevant gov/edu sources"],
            "output_style": "Executive summary followed by bullet-point evidence and numbered citations.",
            "safety_note": (
                "Search is routed through a privacy-preserving meta-search layer instead of direct queries "
                "to a single engine, which limits tracking and ad targeting against your question."
            )
        }

    # Normalize fields in case the model returns strings instead of arrays etc.
    if not isinstance(plan.get("steps"), list):
        plan["steps"] = [plan.get("steps", "Gather info")]
    if not isinstance(plan.get("keywords"), list):
        plan["keywords"] = [plan.get("keywords", str(user_question))]
    if not isinstance(plan.get("sources"), list):
        plan["sources"] = [plan.get("sources", "general reputable sources")]

    plan["raw_model_plan"] = raw_plan  # keep original text for debugging if needed

    return plan

def pretty_print_plan(plan: Dict[str, object]) -> None:
    """
    Show the plan in a human-editable way, so the user can approve or edit sources.
    """
    print("\n----- PROPOSED RESEARCH PLAN -----")
    print("\nIntent / Clarification:")
    print(wrap(str(plan["intent"])))
    print("\nSteps:")
    for i, step in enumerate(plan["steps"], 1):
        print(f"  {i}. {step}")
    print("\nSearch Keywords:")
    for kw in plan["keywords"]:
        print(f"  - {kw}")
    print("\nSuggested Source Domains / Source Types:")
    for src in plan["sources"]:
        print(f"  - {src}")
    print("\nPlanned Output Style:")
    print(wrap(str(plan["output_style"])))
    print("\nSafety / Privacy Note:")
    print(wrap(str(plan["safety_note"])))
    print("\nYou may edit the source list above before we continue.")
    print("To accept, press Enter. To override sources, paste a comma-separated list of domains, then Enter.\n")

# -------------------------
# STEP 2: PLAN CONFIRMATION
# -------------------------

def confirm_plan(plan: Dict[str, object]) -> Dict[str, object]:
    """
    Give the user a chance to edit which sources/domains they'd like emphasized.
    """
    pretty_print_plan(plan)

    # user_edit = input("Source override (or blank to accept): ").strip()
    # if user_edit:
    #     new_sources = [s.strip() for s in user_edit.split(",") if s.strip()]
    #     if new_sources:
    #         plan["sources"] = new_sources

    print("\nFinal source emphasis will be:")
    for s in plan["sources"]:
        print(f"  - {s}")
    print()
    return plan

# -------------------------
# STEP 3: EXECUTION
# -------------------------

def execute_plan(user_question: str, plan: Dict[str, object]) -> None:
    """
    1. For each step, run targeted searches using each keyword, biased toward approved sources.
    2. Stream 'agentic thoughts' (what we're doing and why).
    3. Aggregate all results and ask GPT-4.1 for a final structured answer.
    """

    print("----- EXECUTION START -----\n")

    all_hits: List[Dict[str,str]] = []

    # We'll stream reasoning as plain text. This is not hidden chain-of-thought;
    # it's explicit operational narration of what we're doing.
    for step_i, step in enumerate(plan["steps"], 1):
        print(f"[Step {step_i}] {step}")
        for kw in plan["keywords"]:
            print(f"  - Searching for: {kw}")
            print(f"    (focusing on sources: {', '.join(plan['sources'])})")

            # Run search
            hits = web_search(
                query=kw,
                max_results=6,
                site_filters=plan["sources"]
            )
            print(f"    Retrieved {len(hits)} candidate results.")
            # Show a short preview of top 3
            for idx, h in enumerate(hits[:3], 1):
                preview = h['body'][:200].replace("\n", " ")
                print(f"      [{idx}] {h['title']} :: {h['link']}")
                print(f"          {preview}...")
            all_hits.extend(hits)

        print()

    # Deduplicate results by URL to reduce noise
    dedup = {}
    for h in all_hits:
        url = h["link"]
        if url not in dedup:
            dedup[url] = h
    unique_hits = list(dedup.values())

    # Build evidence context for GPT
    evidence_chunks = []
    for i, h in enumerate(unique_hits, 1):
        evidence_chunks.append(
            f"[{i}] Title: {h['title']}\n"
            f"Snippet: {h['body']}\n"
            f"URL: {h['link']}\n"
        )
    evidence_block = "\n".join(evidence_chunks)

    # Ask GPT-4.1 to synthesize final answer.
    synthesis_prompt = f"""
You are a transparent research analyst.

User Question:
{user_question}

Intent / Clarification:
{plan['intent']}

Desired Output Style:
{plan['output_style']}

Instructions:
1. Answer the user's question directly and accurately.
2. Use an "Executive Summary" section first (3-6 tight bullet points).
3. Follow with "Detailed Findings", organized by subtopic.
4. Add a "Sources" section listing numbered sources that back up key claims.
5. Every factual claim that depends on the web evidence below should map to one or more [n] refs.
6. If sources disagree, say so.

Evidence:
{evidence_block}
"""

    print("Asking the model to synthesize final answer...\n")
    final_answer = gpt_chat(
        model=MODEL_REASONING,
        system_msg="You write well-structured research briefings with explicit citations.",
        user_msg=synthesis_prompt,
        temperature=0.3
    )

    print("----- FINAL ANSWER -----\n")
    print(final_answer)
    print("\n----- END -----\n")


# -------------------------
# MAIN LOOP
# -------------------------

def run_agent(user_prompt, gpt_key):
    """
    Allows multiple questions in one Colab cell execution.
    Type 'exit' to stop.
    """
    # while True:
    user_question = user_prompt.strip()
        # if user_question.lower() in ["exit", "quit", "q"]:
        #     print("Session ended.")
        #     break
    get_api_key(gpt_key)
        # STEP 1: draft plan
    plan = generate_search_plan(user_question)

        # STEP 2: confirm / edit sources
    plan = confirm_plan(plan)

        # STEP 3: execute & answer
    execute_plan(user_question, plan)


# -------------------------
# RUN
# -------------------------

# After the cell finishes defining everything, call run_agent()
# Uncomment the line below to start interactive mode automatically






# You can add helper functions below
# print(main("how many dogs in a kanal house, do I need to keep in chandigarh and under what condition", [465]))