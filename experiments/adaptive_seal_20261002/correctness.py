"""Compare readable cold prompts with partial CPU prefix restores."""

import json
from concurrent.futures import ThreadPoolExecutor


def check(directory, policies, configure, glm, prime_options, prefix_len=11776):
    from transformers import AutoTokenizer

    directory.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(glm.MODEL), local_files_only=True)
    requests, primes = [], []
    for index in range(16):
        prefix = tokenizer.encode(
            f"Document {index:06d}. "
            + "The research team recorded river flow and rainfall every day. "
            * (prefix_len // 8 + 200),
            add_special_tokens=False,
        )[:prefix_len]
        tail = tokenizer.encode(
            "The measurements were recorded in the research notebook. " * 40
            + "\nQuestion: What two quantities did the team measure?\nAnswer:",
            add_special_tokens=False,
        )
        prime_tail = tokenizer.encode(
            "Priming the document cache. " * 40, add_special_tokens=False
        )
        payload = dict(
            model="glm53-compat",
            prompt=prefix + tail,
            max_tokens=5,
            ignore_eos=True,
            temperature=0,
            seed=55,
            add_special_tokens=False,
        )
        requests.append(payload)
        primes.append(dict(payload, prompt=prefix + prime_tail, max_tokens=1))

    def infer(payload):
        return json.loads(glm.call("/v1/completions", payload))

    configure(directory, prime_options, "cold_setting")
    glm.reset_cache(directory, "cold_reset", external=True)
    with ThreadPoolExecutor(max_workers=16) as pool:
        cold = list(pool.map(infer, requests))
    glm.save(directory / "cold.json", dict(requests=requests, responses=cold))
    expected_answer = "river flow and rainfall."
    if any(r["choices"][0]["text"].strip() != expected_answer for r in cold):
        raise RuntimeError("cold probe did not produce the known five-token answer")
    summaries = {}
    for name, options in policies.items():
        target = directory / name
        target.mkdir(exist_ok=True)
        configure(target, prime_options, "prime_setting")
        glm.reset_cache(target, "reset_all", external=True)
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(infer, primes))
        glm.reset_cache(target, "reset_gpu", external=False)
        configure(target, options, "policy_setting")
        with ThreadPoolExecutor(max_workers=16) as pool:
            restored = list(pool.map(infer, requests))
        glm.save(target / "restored.json", restored)
        matches = sum(
            a["choices"][0]["text"] == b["choices"][0]["text"]
            for a, b in zip(cold, restored)
        )
        cached = [
            r["usage"]["prompt_tokens_details"]["cached_tokens"] for r in restored
        ]
        summaries[name] = dict(
            text_matches=matches,
            requests=16,
            answer=expected_answer,
            completion_tokens=5,
            cached_tokens=cached,
            passed=matches == 16
            and cached == [prefix_len] * 16
            and all(r["usage"]["completion_tokens"] == 5 for r in restored),
        )
        glm.save(directory / "summary.json", summaries)
        print("CORRECTNESS", name, summaries[name], flush=True)
        if not summaries[name]["passed"]:
            raise RuntimeError(f"cold/CPU restore correctness failed for {name}")
    configure(directory, prime_options, "drain_setting")


def check_distinct(
    directory, policies, configure, glm, prime_options, prefix_len=11776
):
    """Check request identity using different facts in otherwise similar prefixes."""
    from transformers import AutoTokenizer

    directory.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(glm.MODEL), local_files_only=True)
    pairs = [
        ("temperature", "pressure"),
        ("speed", "distance"),
        ("voltage", "current"),
        ("height", "weight"),
        ("rainfall", "humidity"),
        ("length", "width"),
        ("mass", "volume"),
        ("latitude", "longitude"),
        ("frequency", "amplitude"),
        ("density", "viscosity"),
        ("wavelength", "intensity"),
        ("conductivity", "resistance"),
        ("depth", "turbidity"),
        ("altitude", "velocity"),
        ("acidity", "salinity"),
        ("diameter", "circumference"),
    ]
    requests, primes, expected = [], [], []
    for index in range(16):
        first, second = pairs[index % len(pairs)]
        sentence = f"The research team measured {first} and {second} every day. "
        prefix = tokenizer.encode(
            f"Document {index:06d}. " + sentence * (prefix_len // 8 + 200),
            add_special_tokens=False,
        )[:prefix_len]
        tail = tokenizer.encode(
            "The measurements were recorded in the research notebook. "
            * 40
            + "\nQuestion: What two quantities did this team measure? "
            "Return only their names.\nAnswer:",
            add_special_tokens=False,
        )
        prime_tail = tokenizer.encode(
            "Priming the document cache. " * 40, add_special_tokens=False
        )
        payload = dict(
            model="glm53-compat",
            prompt=prefix + tail,
            max_tokens=24,
            stop=[".", "\n"],
            temperature=0,
            seed=55,
            add_special_tokens=False,
        )
        requests.append(payload)
        primes.append(dict(payload, prompt=prefix + prime_tail, max_tokens=1))
        expected.append((first, second))

    def infer(payload):
        return json.loads(glm.call("/v1/completions", payload))

    def factual_matches(responses):
        return [
            all(word in response["choices"][0]["text"].casefold() for word in pair)
            for response, pair in zip(responses, expected)
        ]

    configure(directory, prime_options, "cold_setting")
    glm.reset_cache(directory, "cold_reset", external=True)
    with ThreadPoolExecutor(max_workers=16) as pool:
        cold = list(pool.map(infer, requests))
    glm.save(
        directory / "cold.json",
        dict(requests=requests, expected=expected, responses=cold),
    )
    if not all(factual_matches(cold)):
        raise RuntimeError(
            "cold distinct-fact probe did not return its known quantities"
        )
    summaries = {}
    for name, options in policies.items():
        target = directory / name
        target.mkdir(exist_ok=True)
        configure(target, prime_options, "prime_setting")
        glm.reset_cache(target, "reset_all", external=True)
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(infer, primes))
        glm.reset_cache(target, "reset_gpu", external=False)
        configure(target, options, "policy_setting")
        with ThreadPoolExecutor(max_workers=16) as pool:
            restored = list(pool.map(infer, requests))
        glm.save(target / "restored.json", restored)
        cached = [
            r["usage"]["prompt_tokens_details"]["cached_tokens"] for r in restored
        ]
        factual = factual_matches(restored)
        summaries[name] = dict(
            requests=len(requests),
            expected=expected,
            factual_matches=sum(factual),
            exact_cold_matches=sum(
                a["choices"][0]["text"] == b["choices"][0]["text"]
                for a, b in zip(cold, restored)
            ),
            cached_tokens=cached,
            passed=all(factual) and cached == [prefix_len] * len(requests),
        )
        glm.save(directory / "summary.json", summaries)
        print("DISTINCT_CORRECTNESS", name, summaries[name], flush=True)
        if not summaries[name]["passed"]:
            raise RuntimeError(f"request identity correctness failed for {name}")
    configure(directory, prime_options, "drain_setting")
