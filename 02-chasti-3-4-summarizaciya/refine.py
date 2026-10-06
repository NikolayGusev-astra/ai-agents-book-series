"""Generic refine.py - incremental summarization with a running draft.

No provider bindings: any OpenAI-compatible endpoint. Reproduces the refine
strategy from the article: a draft lives through the whole series, each step
sees draft + next chunk.

Known failure mode (see article): naive "draft = full previous answer" drifts
to recency bias - the final draft remembers the tail and forgets the beginning.
The --structured flag keeps the draft as a numbered topic list instead of prose,
which is the first-line mitigation.

Usage:
  python refine.py --input doc.txt --base-url http://localhost:1234/v1 \\
      --model yandexgpt-5-lite-8b-instruct [--chunk 6000] [--structured] \\
      [--out summary_refine.txt]
"""
import argparse, json, time, urllib.request


def llm(base_url, api_key, model, system, user, max_tokens=800, temperature=0.2):
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions", data=payload,
        headers={"Content-Type": "application/json",
                  "Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--base-url", default="http://localhost:1234/v1")
    ap.add_argument("--api-key", default="NO-KEY-NEEDED")
    ap.add_argument("--model", required=True)
    ap.add_argument("--chunk", type=int, default=6000)
    ap.add_argument("--min-chunk", type=int, default=1500)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--structured", action="store_true",
                    help="keep draft as a numbered topic list (recency-bias mitigation)")
    ap.add_argument("--out", default="summary_refine.txt")
    a = ap.parse_args()

    text = open(a.input, encoding="utf-8").read()
    raw = [text[i:i + a.chunk] for i in range(0, len(text), a.chunk)]
    chunks = [c for c in raw if len(c) >= a.min_chunk]
    print(f"{len(chunks)} chunks x ~{a.chunk} chars")

    if a.structured:
        sys_refine = (
            "You are a technical editor maintaining a numbered topic list that summarizes "
            "a book. You get the current list and a new fragment. Add new topics as numbered "
            "items, extend existing ones if the fragment adds facts. NEVER delete or reword "
            "existing items - only append or extend. If the fragment adds nothing new, return "
            "the list unchanged. Answer in Russian.")
    else:
        sys_refine = (
            "You are a technical editor. You get the current draft summary and a new book "
            "fragment. Extend the draft with the essential content of the fragment (2-4 "
            "sentences). If the fragment adds nothing new, return the draft unchanged. "
            "Answer in Russian.")

    t0 = time.time()
    draft = ""
    no_change = 0
    for i, ch in enumerate(chunks, 1):
        out = llm(a.base_url, a.api_key, a.model, sys_refine,
                  f"Current draft:\\n{draft if draft else '(empty)'}\\n\\nNew fragment {i}/{len(chunks)}:\\n\\n{ch}",
                  max_tokens=a.max_tokens).strip()
        if out == draft:
            no_change += 1
        draft = out
        print(f"  refine {i}/{len(chunks)} done (len={len(draft)})")
    print(f"Refine: {time.time() - t0:.1f}s, {len(chunks)} calls, {no_change} no-change steps")

    open(a.out, "w", encoding="utf-8").write(draft)
    print("saved:", a.out)


if __name__ == "__main__":
    main()
