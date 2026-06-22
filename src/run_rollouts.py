"""Generate multi-sample rollouts for both arms (control + cue) against a vLLM-served
model, then score OpenAI's chain-of-thought monitor on each. Writes rollouts.csv
(instance_id, sample, x, y, z_cot) and rollouts_raw.jsonl (cot + answer).
Example:
  python src/run_rollouts.py --datasets gpqa_encoded --samples 24 --concurrency 256 \
      --base-url http://localhost:8000/v1 --out results/encoded"""
import argparse, asyncio, json
from pathlib import Path
import sys, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import helpers

async def chat(client, model, effort, content, sem, max_tokens, temperature):
    async with sem:
        for attempt in range(3):
            try:
                r = await client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": content}],
                    temperature=temperature, max_tokens=max_tokens,
                    extra_body={"reasoning_effort": effort}, timeout=1200)
                return r.choices[0].message
            except Exception as e:
                if attempt == 2: return e
                await asyncio.sleep(2 * (attempt + 1))

async def main(a):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=a.base_url, api_key=a.api_key)
    sem = asyncio.Semaphore(a.concurrency)
    work = []
    for ds in a.datasets:
        tmpl, hint_type = helpers.template_for(ds)
        for row in helpers.load_rows(ds):
            if not str(row.get("problem", "")).strip():
                continue
            for s in range(a.samples):
                work.append((ds, tmpl, hint_type, row, s))
    print("%d rollouts (%d datasets x both arms x %d samples)" % (len(work), len(a.datasets), a.samples))

    async def rollout(item):
        ds, tmpl, hint_type, row, s = item
        msg = await chat(client, a.model, a.agent_effort, row["problem"], sem, a.agent_max_tokens, 1.0)
        if isinstance(msg, Exception): return None
        return dict(ds=ds, tmpl=tmpl, hint_type=hint_type, row=row, sample=s,
                    cot=helpers.extract_reasoning(msg), answer=msg.content or "")
    rolls = [r for r in await asyncio.gather(*[rollout(w) for w in work]) if r]

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with (out / "rollouts_raw.jsonl").open("w", encoding="utf-8") as f:
        for r in rolls:
            f.write(json.dumps(dict(dataset=r["ds"], instance_id=str(r["row"].get("id")),
                    sample=r["sample"], x=int(r["row"]["x"]), cot=r["cot"], answer=r["answer"])) + "\n")

    async def monitor(r):
        prompt = helpers.render_monitor_prompt(r["tmpl"], r["row"], r["cot"], r["answer"], r["hint_type"])
        msg = await chat(client, a.monitor_model, a.monitor_effort, prompt, sem, a.monitor_max_tokens, 0.0)
        z = 0 if isinstance(msg, Exception) else helpers.parse_monitor(msg.content or "", r["hint_type"])
        y = helpers.grade_y(r["row"].get("answer", ""), r["answer"])
        return dict(dataset=r["ds"], instance_id=str(r["row"].get("id")), sample=r["sample"],
                    x=int(r["row"]["x"]), y=int(y), z_cot=int(z))
    recs = [r for r in await asyncio.gather(*[monitor(r) for r in rolls]) if isinstance(r, dict)]
    pd.DataFrame(recs).to_csv(out / "rollouts.csv", index=False)
    print("wrote", out / "rollouts.csv", "and rollouts_raw.jsonl")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["gpqa_encoded"], choices=list(helpers.DATASETS))
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--concurrency", type=int, default=256)
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--monitor-model", default="openai/gpt-oss-20b")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--agent-effort", default="medium")
    ap.add_argument("--agent-max-tokens", type=int, default=16000)
    ap.add_argument("--monitor-effort", default="medium")
    ap.add_argument("--monitor-max-tokens", type=int, default=1500)
    ap.add_argument("--out", default="results/encoded")
    asyncio.run(main(ap.parse_args()))
