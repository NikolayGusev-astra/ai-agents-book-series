"""Generic mapreduce.py - summarization of a long document by chunks.

No bindings to specific providers: any OpenAI-compatible endpoint (base_url + key).
Reproduces the MapReduce strategy from the article:
  split -> map (summarize each chunk) -> reduce (merge the summaries).

Usage:
  python mapreduce.py --input doc.txt --base-url http://localhost:1234/v1 \
      --model yandexgpt-5-lite-8b-instruct [--api-key NO-KEY-NEEDED] \
      [--chunk 6000] [--out summary.txt]
"""
import argparse, json, time, urllib.request


def llm(base_url, api_key, model, system, user, max_tokens=700, temperature=0.2):
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
    ap.add_argument("--chunk", type=int, default=6000, help="chunk size, chars")
    ap.add_argument("--min-chunk", type=int, default=1500, help="drop tail shorter than this")
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--out", default="summary_mapreduce.txt")
    a = ap.parse_args()

    text = open(a.input, encoding="utf-8").read()
    raw = [text[i:i + a.chunk] for i in range(0, len(text), a.chunk)]
    chunks = [c for c in raw if len(c) >= a.min_chunk]
    print(f"{len(chunks)} chunks x ~{a.chunk} chars")

    sys_map = ("You are a technical editor. Summarize the book fragment briefly: "
               "3-5 sentences, facts and mechanisms only. Answer in Russian.")
    sys_reduce = ("You are a technical editor. Merge the partial summaries into one "
                  "coherent summary of 2-3 paragraphs, drop duplicates. Answer in Russian.")

    t0 = time.time()
    maps = []
    for i, ch in enumerate(chunks, 1):
        out = llm(a.base_url, a.api_key, a.model, sys_map, f"Fragment {i}/{len(chunks)}:\n\n{ch}")
        maps.append(out)
        print(f"  map {i}/{len(chunks)} done")
    joined = "\n\n".join(f"- {m}" for m in maps)
    final = llm(a.base_url, a.api_key, a.model, sys_reduce, joined, max_tokens=a.max_tokens + 200)
    print(f"MapReduce: {time.time() - t0:.1f}s, {len(chunks) + 1} calls")

    open(a.out, "w", encoding="utf-8").write(final)
    print("saved:", a.out)


if __name__ == "__main__":
    main()
