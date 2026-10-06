"""Эксперимент главы 2 цикла: MapReduce vs Refine на yandexgpt-5-lite-8b (LM Studio :1234).

Вход: 10 разделов книги AI Agents and Applications (~83k символов).
Обе стратегии решают одну задачу: саммари о том, как выбирать LLM и программировать промпты.
Замер: время, число вызовов LLM, размер итога.
"""
import urllib.request, json, time, re

BASE = "http://127.0.0.1:1234/v1/chat/completions"
MODEL = "yandexgpt-5-lite-8b-instruct"
# без прокси (корпоративный ловит localhost)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

def llm(system, user, max_tokens=700):
    payload = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(BASE, data=payload, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with opener.open(req, timeout=300) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    return d["choices"][0]["message"]["content"], dt

# ---- input
text = open(r"C:/Work/Assist/AI-Study/02-части-3-4-суммаризация/experiment-source.txt", encoding="utf-8").read()
CHUNK = 6000
raw_chunks = [text[i:i+CHUNK] for i in range(0, len(text), CHUNK)]
chunks = [c for c in raw_chunks if len(c) > 1500]  # хвост-огрызок выбрасываем
print(f"chunks: {len(chunks)} x ~{CHUNK} chars")

SYS_MAP = "Ты технический редактор. Кратко резюмируй фрагмент книги: 3-5 предложений, только факты и механизмы, без воды. Русский язык."
SYS_REDUCE = "Ты технический редактор. Склей частные резюме фрагментов книги в единое связное саммари: 2-3 абзаца. Убирай повторы. Русский язык."
SYS_REFINE = ("Ты технический редактор. У тебя есть текущий черновик саммари книги и новый фрагмент. "
              "Дополни черновик существенным из фрагмента (2-4 предложения). Если фрагмент не даёт нового - "
              "верни черновик без изменений. Русский язык.")

results = {}

# ---- MapReduce
t0 = time.time()
maps = []
for i, ch in enumerate(chunks):
    out, dt = llm(SYS_MAP, f"Фрагмент {i+1}/{len(chunks)}:\n\n{ch}")
    maps.append((out, dt))
    print(f"  map {i+1}/{len(chunks)}: {dt:.1f}s")
joined = "\n\n".join(f"- {m[0]}" for m in maps)
mr_final, mr_reduce_dt = llm(SYS_REDUCE, joined, max_tokens=900)
mr_time = time.time() - t0
results["mapreduce"] = {
    "time_s": round(mr_time, 1),
    "llm_calls": len(chunks) + 1,
    "map_total_s": round(sum(d for _, d in maps), 1),
    "reduce_s": round(mr_reduce_dt, 1),
    "final": mr_final,
}
print(f"MapReduce: {mr_time:.1f}s total, {len(chunks)+1} calls")

# ---- Refine
t0 = time.time()
draft = ""
refine_steps = 0
skipped = 0
for i, ch in enumerate(chunks):
    out, dt = llm(SYS_REFINE,
                  f"Текущий черновик:\n{draft if draft else '(пусто)'}\n\nНовый фрагмент {i+1}/{len(chunks)}:\n\n{ch}",
                  max_tokens=800)
    # detect no-change
    if out.strip() == draft.strip():
        skipped += 1
    draft = out.strip()
    refine_steps += 1
    print(f"  refine {i+1}/{len(chunks)}: {dt:.1f}s")
rf_time = time.time() - t0
results["refine"] = {
    "time_s": round(rf_time, 1),
    "llm_calls": refine_steps,
    "no_change_steps": skipped,
    "final": draft,
}
print(f"Refine: {rf_time:.1f}s total, {refine_steps} calls, {skipped} no-change")

with open(r"C:/Work/Assist/AI-Study/02-части-3-4-суммаризация/experiment-results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\n=== MapReduce final ===\n", mr_final[:500])
print("\n=== Refine final ===\n", draft[:500])
