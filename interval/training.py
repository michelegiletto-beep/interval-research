"""Local supervised LoRA fit, with an untouched synthetic language evaluation set."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import json
import random
import torch
from .planner import ROOT, RECIPES, render_prompt, score_candidates

TRAIN_TEMPLATES = [
    "Please {action}. Use seed {seed} for this research run.",
    "My next experiment should {action}. Keep its evidence local.",
    "For project {project}, I want to {action}.",
    "Can you {action}? Use independent calibration and test observations.",
    "I need a reproducible task: {action}.",
    "The requested research task is to {action}, using the current dataset."]
TEST_TEMPLATES = [
    "Arrange a documented experiment to {action}. Project: {project}; seed: {seed}.",
    "The lab would like to {action}. Which available workflow should run?"]
ACTIONS = {
    "compare": ["compare classical, neural and quantum estimators", "benchmark every background estimator", "run the full estimator comparison", "evaluate the available methods side by side"],
    "tails": ["sweep marginal tail strength", "study heavy-tailed observation sensitivity", "compare weak, medium and strong distribution tails", "repeat the experiment for three tail strengths"],
    "measurements": ["study finite measurement counts and readout errors", "compare exact sampling with noisy measurements", "measure shot noise sensitivity", "inspect the effect of measurement-bit flips"],
    "neural": ["train and evaluate the conditional neural network", "fit a neural background estimator", "assess the learned neural conditional mean", "train the neural model on local observations"]}


def instruction_data():
    train, test = [], []
    rng = random.Random(8173)
    for recipe, actions in ACTIONS.items():
        for template in TRAIN_TEMPLATES:
            for action in actions:
                request = template.format(action=action, seed=rng.randrange(1000), project=rng.choice(["Tissue study", "Infrastructure survey", "Client workspace"]))
                train.append({"request": request, "recipe": recipe})
        for template in TEST_TEMPLATES:
            for action in actions:
                request = template.format(action=action, seed=rng.randrange(1000), project="External evaluation")
                test.append({"request": request, "recipe": recipe})
    assert not {r["request"] for r in train} & {r["request"] for r in test}
    rng.shuffle(train)
    return train, test


def evaluate(tokenizer, model, examples):
    records = []
    for row in examples:
        scores = score_candidates(tokenizer, model, row["request"])
        prediction = max(scores, key=scores.get)
        records.append({**row, "prediction": prediction, "correct": prediction == row["recipe"]})
    return {"accuracy": sum(r["correct"] for r in records) / len(records), "n": len(records), "records": records}


def train_planner(steps=120, progress=None, train_path=None, test_path=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from .planner import unload_model
    if not 10 <= steps <= 1000:
        raise ValueError("Choose 10–1,000 local adaptation steps")
    if bool(train_path) != bool(test_path):
        raise ValueError("Provide both independent training and evaluation instruction files")
    if train_path:
        def read_instructions(path):
            rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
            if not 8 <= len(rows) <= 10000:
                raise ValueError("Each instruction file must contain 8–10,000 examples")
            if any(set(r) != {"request", "recipe"} or not isinstance(r["request"], str) or not 3 <= len(r["request"]) <= 1500 or r["recipe"] not in RECIPES for r in rows):
                raise ValueError("Instruction rows require request text and an allowed recipe")
            if set(r["recipe"] for r in rows) != set(RECIPES):
                raise ValueError("Each instruction split must cover all four recipes")
            return rows
        custom_train, custom_test = read_instructions(train_path), read_instructions(test_path)
        if {r["request"].strip().casefold() for r in custom_train} & {r["request"].strip().casefold() for r in custom_test}:
            raise ValueError("Training and evaluation instructions overlap")
    unload_model()
    emit = progress or (lambda message, detail: print(message, flush=True))
    torch.manual_seed(8173)
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start = perf_counter()
    base = ROOT / "models" / "base"
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, torch_dtype=torch.float32, attn_implementation="eager").to(device)
    train, test = (custom_train, custom_test) if train_path else instruction_data()
    emit("Evaluating the unadapted language model", {"examples": len(test), "device": device})
    model.eval()
    before = evaluate(tokenizer, model, test)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05,
                           target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    model.config.use_cache = False
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=4e-4)
    encoded = []
    for row in train:
        prompt = render_prompt(tokenizer, row["request"])
        answer = json.dumps({"recipe": row["recipe"]}, separators=(",", ":")) + tokenizer.eos_token
        prefix = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        ids = tokenizer(prompt + answer, add_special_tokens=False)["input_ids"]
        labels = [-100] * len(prefix) + ids[len(prefix):]
        encoded.append((ids, labels))
    model.train()
    losses = []
    for step in range(steps):
        batch = [encoded[(step * 4 + i) % len(encoded)] for i in range(4)]
        width = max(len(item[0]) for item in batch)
        ids = torch.full((len(batch), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
        labels = torch.full_like(ids, -100)
        mask = torch.zeros_like(ids)
        for i, (tokens, targets) in enumerate(batch):
            ids[i, :len(tokens)] = torch.tensor(tokens, device=device)
            labels[i, :len(targets)] = torch.tensor(targets, device=device)
            mask[i, :len(tokens)] = 1
        optimizer.zero_grad()
        loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 10 == 0:
            emit(f"Local adaptation step {step + 1}/{steps}", {"loss": losses[-1]})
    model.eval()
    model.config.use_cache = True
    emit("Evaluating the adapted language model on held-out wording", {})
    after = evaluate(tokenizer, model, test)
    destination = ROOT / "models" / "planner"
    model.save_pretrained(destination, safe_serialization=True)
    adapter_config_path = destination / "adapter_config.json"
    portable_config = json.loads(adapter_config_path.read_text())
    portable_config["base_model_name_or_path"] = "HuggingFaceTB/SmolLM2-135M-Instruct"
    portable_config["revision"] = json.loads((ROOT / "models" / "base-model.json").read_text())["revision"]
    adapter_config_path.write_text(json.dumps(portable_config, indent=2))
    tokenizer.save_pretrained(destination)
    adapter_hash = sha256((destination / "adapter_model.safetensors").read_bytes()).hexdigest()
    data_dir = ROOT / "datasets" / "planner"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("test", test)):
        (data_dir / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    receipt = {"created_utc": datetime.now(timezone.utc).isoformat(), "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
               "base_revision": json.loads((ROOT / "models" / "base-model.json").read_text())["revision"],
               "license": "apache-2.0", "method": "LoRA supervised synthetic instruction adaptation",
               "steps": steps, "batch_size": 4, "learning_rate": 4e-4, "rank": 8,
               "device": device, "hardware": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
               "train_examples": len(train), "test_examples": len(test), "base_evaluation": before,
               "data_origin": "user-supplied-local-instructions" if train_path else "synthetic-template-held-out",
               "adapted_evaluation": after, "training_loss": losses, "seconds": perf_counter() - start,
               "adapter_sha256": adapter_hash, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
               "limits": ["Synthetic held-out phrasing of four known recipes; no evidence of general reasoning or autonomous discovery.",
                          "Candidate likelihood ranking always returns a schema-valid recipe by construction.",
                          "Accuracy measures intent selection, not scientific validity of an experiment."]}
    (ROOT / "models" / "training-receipt.json").write_text(json.dumps(receipt, indent=2))
    emit("Saved local adapter and training receipt", {"accuracy": after["accuracy"], "seconds": receipt["seconds"]})
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return receipt
