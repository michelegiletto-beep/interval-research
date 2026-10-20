"""Embedded open-weight language model selects a bounded, executable recipe.

Selection uses mean suffix-token conditional log likelihood, not substring
extraction or unconstrained tool calls. Single-user local execution is offered;
there are no shell, network, publication or external-action tools.
"""
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import json
import threading
import torch

ROOT = Path(__file__).resolve().parents[1]
RECIPES = {
    "compare": {"title": "Compare estimators", "purpose": "Compare classical, neural and quantum conditional estimates with held-out calibration.",
                "stages": ["inspect", "fit", "calibrate", "evaluate", "archive"], "runs": 1},
    "tails": {"title": "Tail sensitivity", "purpose": "Repeat a frozen protocol at three marginal tail strengths.",
              "stages": ["inspect", "fit", "calibrate", "evaluate", "archive"], "runs": 3},
    "measurements": {"title": "Measurement sensitivity", "purpose": "Compare exact probabilities, finite measurements and terminal bit-flip noise.",
                     "stages": ["inspect", "fit", "calibrate", "evaluate", "archive"], "runs": 3},
    "neural": {"title": "Train and assess the neural estimator", "purpose": "Fit the conditional network and compare it against the same reference estimators.",
               "stages": ["inspect", "fit", "calibrate", "evaluate", "archive"], "runs": 1}}
SYSTEM = "Select the research recipe requested by the user. Reply with exactly one JSON object. Allowed recipes: compare, tails, measurements, neural."
_lock = threading.RLock()
_cache = None


def render_prompt(tokenizer, request):
    return tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": request}],
                                         tokenize=False, add_generation_prompt=True)


def load_model(use_adapter=True):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = ROOT / "models" / "base"
    if not (base / "config.json").exists():
        raise FileNotFoundError("Embedded model is not installed. Run python scripts/download_model.py, then train-planner.")
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, torch_dtype=torch.float32,
                                                attn_implementation="eager")
    adapter = ROOT / "models" / "planner"
    if use_adapter and (adapter / "adapter_config.json").exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False, torch_device="cpu")
    model.eval()
    return tokenizer, model


def score_candidates(tokenizer, model, request):
    prompt = render_prompt(tokenizer, request)
    # Each candidate has its own exact token boundary; include its complete answer.
    scores = {}
    for recipe in RECIPES:
        answer = json.dumps({"recipe": recipe}, separators=(",", ":"))
        p = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full = tokenizer(prompt + answer, add_special_tokens=False, return_tensors="pt")
        labels = full["input_ids"].clone()
        labels[:, :len(p)] = -100
        full = {k: v.to(model.device) for k, v in full.items()}
        labels = labels.to(model.device)
        with torch.inference_mode():
            loss = model(**full, labels=labels).loss
        scores[recipe] = -float(loss)
    return scores


def plan(request, manual_recipe=None):
    global _cache
    start = perf_counter()
    if manual_recipe is not None:
        if manual_recipe not in RECIPES:
            raise ValueError("Unknown manual recipe")
        recipe, origin, scores = manual_recipe, "explicit-user-selection", None
    else:
        with _lock:
            if _cache is None:
                _cache = load_model()
            scores = score_candidates(*_cache, request)
        recipe, origin = max(scores, key=scores.get), "embedded-language-model-likelihood"
    training_path = ROOT / "models" / "training-receipt.json"
    receipt = json.loads(training_path.read_text()) if training_path.exists() else None
    return {"recipe": recipe, **RECIPES[recipe], "origin": origin, "candidate_scores": scores,
            "seconds": perf_counter() - start, "request_sha256": sha256(request.encode()).hexdigest(),
            "model": receipt.get("model") if receipt else "HuggingFaceTB/SmolLM2-135M-Instruct (base, not locally adapted)",
            "adapter_sha256": receipt.get("adapter_sha256") if receipt else None,
            "scope": "Four predefined research recipes. No unrestricted tool use or external actions."}


def unload_model():
    global _cache
    with _lock:
        _cache = None
