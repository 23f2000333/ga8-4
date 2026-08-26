import json
import math
import re
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

SAFE_MAX = 9007199254740991
SHA64 = re.compile(r"^[0-9a-f]{64}$")
SHA40 = re.compile(r"^[0-9a-f]{40}$")
INTERVENTIONS = ["prompt_only", "retrieval", "lora", "qlora"]

CHOOSE_CODES = {
    "INVALID_INPUT",
    "UNAVAILABLE",
    "QUALITY_FLOOR",
    "FRESHNESS_REQUIRED",
    "LATENCY_LIMIT",
    "MEMORY_LIMIT",
    "DATA_LIMIT",
    "COST_LIMIT",
}

REPAIR_CODES = {
    "INVALID_TOKEN",
    "INVALID_PARAMETER",
    "CHAT_TEMPLATE_COUNT",
    "INFERENCE_MODE",
    "FULL_MODEL_ARTIFACT",
    "ADAPTER_FILE_SET",
    "INCOMPLETE_CHECKPOINT",
    "MUTABLE_BASE_REVISION",
    "LINEAGE_MISMATCH",
    "EFFECTIVE_BATCH_MISMATCH",
    "EVAL_LEAKAGE",
    "EVAL_DROPOUT_ACTIVE",
    "RESUME_DIVERGENCE",
}


def bkey(x: str) -> bytes:
    return x.encode("utf-8")


def finite(x):
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(x)
    )


def safe_int(x):
    return (
        isinstance(x, int)
        and not isinstance(x, bool)
        and 0 <= x <= SAFE_MAX
    )


def positive_safe_int(x):
    return safe_int(x) and x > 0


def nonempty_string(x):
    return isinstance(x, str) and len(x) > 0


def unique_strings(xs):
    return (
        isinstance(xs, list)
        and all(isinstance(x, str) and x for x in xs)
        and len(xs) == len(set(xs))
    )


def sorted_codes(codes):
    return sorted(set(codes), key=bkey)


def strict_json(raw):
    return json.loads(
        raw.decode("utf-8"),
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
    )


def choose_policy_valid(p):
    if not isinstance(p, dict):
        return False

    required = {
        "minQuality",
        "freshnessRequired",
        "maxLatencyMs",
        "maxMemoryMb",
        "maxLabeledExamples",
        "maxTotalCost",
        "horizonRequests",
    }

    if set(p.keys()) != required:
        return False

    if not finite(p["minQuality"]) or not 0 <= p["minQuality"] <= 1:
        return False

    if not isinstance(p["freshnessRequired"], bool):
        return False

    for k in ("maxLatencyMs", "maxMemoryMb", "maxTotalCost"):
        if not finite(p[k]) or p[k] < 0:
            return False

    if not safe_int(p["maxLabeledExamples"]):
        return False
    if not safe_int(p["horizonRequests"]):
        return False

    return True


def candidate_valid_shape(c):
    if not isinstance(c, dict):
        return False

    required = {
        "name",
        "available",
        "quality",
        "freshness",
        "latencyMs",
        "memoryMb",
        "labeledExamples",
        "oneTimeCost",
        "recurringCost",
    }

    if set(c.keys()) != required:
        return False

    if not isinstance(c["name"], str):
        return False
    if not isinstance(c["available"], bool):
        return False
    if not finite(c["quality"]) or not 0 <= c["quality"] <= 1:
        return False
    if not isinstance(c["freshness"], bool):
        return False

    for k in ("latencyMs", "memoryMb", "oneTimeCost", "recurringCost"):
        if not finite(c[k]) or c[k] < 0:
            return False

    if not safe_int(c["labeledExamples"]):
        return False

    return True


def choose(payload):
    codes = {name: [] for name in INTERVENTIONS}
    total_costs = {name: None for name in INTERVENTIONS}

    if not isinstance(payload, dict):
        return {
            "selected": None,
            "eligible": [],
            "totalCosts": total_costs,
            "reasonCodes": {
                name: ["INVALID_INPUT"] for name in INTERVENTIONS
            },
        }

    required = {"operation", "policy", "candidates"}
    if set(payload.keys()) != required or payload.get("operation") != "choose":
        return {
            "selected": None,
            "eligible": [],
            "totalCosts": total_costs,
            "reasonCodes": {
                name: ["INVALID_INPUT"] for name in INTERVENTIONS
            },
        }

    policy = payload["policy"]
    candidates = payload["candidates"]

    if not choose_policy_valid(policy):
        return {
            "selected": None,
            "eligible": [],
            "totalCosts": total_costs,
            "reasonCodes": {
                name: ["INVALID_INPUT"] for name in INTERVENTIONS
            },
        }

    if not isinstance(candidates, list) or len(candidates) != 4:
        return {
            "selected": None,
            "eligible": [],
            "totalCosts": total_costs,
            "reasonCodes": {
                name: ["INVALID_INPUT"] for name in INTERVENTIONS
            },
        }

    by_name = {}
    valid_all = True

    for c in candidates:
        if not candidate_valid_shape(c):
            valid_all = False
            continue
        if c["name"] in by_name:
            valid_all = False
            continue
        by_name[c["name"]] = c

    if (
        not valid_all
        or set(by_name.keys()) != set(INTERVENTIONS)
    ):
        return {
            "selected": None,
            "eligible": [],
            "totalCosts": total_costs,
            "reasonCodes": {
                name: ["INVALID_INPUT"] for name in INTERVENTIONS
            },
        }

    eligible = []

    for name in INTERVENTIONS:
        c = by_name[name]
        local = []

        total = round(
            c["oneTimeCost"]
            + policy["horizonRequests"] * c["recurringCost"],
            12,
        )
        total_costs[name] = total

        if not c["available"]:
            local.append("UNAVAILABLE")

        if c["quality"] < policy["minQuality"]:
            local.append("QUALITY_FLOOR")

        if policy["freshnessRequired"] and not c["freshness"]:
            local.append("FRESHNESS_REQUIRED")

        if c["latencyMs"] > policy["maxLatencyMs"]:
            local.append("LATENCY_LIMIT")

        if c["memoryMb"] > policy["maxMemoryMb"]:
            local.append("MEMORY_LIMIT")

        if c["labeledExamples"] > policy["maxLabeledExamples"]:
            local.append("DATA_LIMIT")

        if total > policy["maxTotalCost"]:
            local.append("COST_LIMIT")

        codes[name] = sorted_codes(local)

        if not codes[name]:
            eligible.append(name)

    return {
        "selected": eligible[0] if eligible else None,
        "eligible": eligible,
        "totalCosts": total_costs,
        "reasonCodes": codes,
    }


def valid_token(t):
    if not isinstance(t, dict):
        return False

    if set(t.keys()) != {"id", "role", "padding", "text"}:
        return False

    return (
        safe_int(t["id"])
        and t["role"] in {"system", "user", "assistant"}
        and isinstance(t["padding"], bool)
        and isinstance(t["text"], str)
    )


def valid_parameter(p):
    if not isinstance(p, dict):
        return False

    if set(p.keys()) != {"name", "target", "numel"}:
        return False

    return (
        isinstance(p["name"], str)
        and len(p["name"]) > 0
        and isinstance(p["target"], str)
        and len(p["target"]) > 0
        and positive_safe_int(p["numel"])
    )


def repair(payload):
    labels = []
    template_pass = False
    trainable_params = []
    trainable_count = 0
    peft_pass = True
    adapter_files = []
    checkpoint_complete = False
    lineage_pass = True
    eval_isolated = True
    evaluation_deterministic = True
    resume_pass = True
    reasons = []

    required = {
        "operation",
        "tokens",
        "templateApplications",
        "parameters",
        "allowedTargets",
        "inferenceMode",
        "trainRowIds",
        "evalRowIds",
        "dropoutActiveDuringEval",
        "artifactFiles",
        "baseRevision",
        "datasetDigest",
        "codeDigest",
        "configDigest",
        "expectedDigests",
        "microBatch",
        "gradientAccumulation",
        "replicas",
        "expectedEffectiveBatch",
        "checkpoint",
        "uninterruptedWeights",
        "resumedWeights",
        "resumeTolerance",
    }

    if (
        not isinstance(payload, dict)
        or set(payload.keys()) != required
        or payload.get("operation") != "repair"
    ):
        return {
            "labels": [],
            "templatePass": False,
            "trainableParams": [],
            "trainableCount": 0,
            "peftConfigPass": False,
            "adapterFiles": [],
            "checkpointComplete": False,
            "lineagePass": False,
            "evalIsolated": False,
            "evaluationDeterministic": False,
            "resumePass": False,
            "reasonCodes": ["INVALID_TOKEN"],
        }

    tokens = payload["tokens"]

    if not isinstance(tokens, list) or len(tokens) == 0:
        reasons.append("INVALID_TOKEN")
        labels = []
    elif all(valid_token(t) for t in tokens):
        labels = [
            t["id"] if t["role"] == "assistant" and not t["padding"] else -100
            for t in tokens
        ]
    else:
        reasons.append("INVALID_TOKEN")
        labels = [-100 for _ in tokens]

    if payload["templateApplications"] == 1:
        template_pass = True
    else:
        reasons.append("CHAT_TEMPLATE_COUNT")

    if payload["inferenceMode"] is not False:
        reasons.append("INFERENCE_MODE")

    params = payload["parameters"]
    allowed = payload["allowedTargets"]

    params_valid = isinstance(params, list) and all(
        valid_parameter(p) for p in params
    )

    if params_valid:
        names = [p["name"] for p in params]
        if len(names) != len(set(names)):
            params_valid = False

    allowed_valid = (
        isinstance(allowed, list)
        and len(allowed) > 0
        and all(isinstance(x, str) and len(x) > 0 for x in allowed)
        and len(allowed) == len(set(allowed))
    )

    if not params_valid or not allowed_valid:
        reasons.append("INVALID_PARAMETER")
        peft_pass = False
    else:
        allowed_set = set(allowed)

        # Only parameters whose target is explicitly allowed AND whose name
        # is a LoRA A/B weight are trainable.
        selected = [
            p for p in params
            if p["target"] in allowed_set
            and (
                p["name"].endswith(".lora_A.weight")
                or p["name"].endswith(".lora_B.weight")
            )
        ]

        if not selected:
            reasons.append("INVALID_PARAMETER")
            peft_pass = False
        else:
            selected.sort(key=lambda p: bkey(p["name"]))

            running_count = 0
            overflow = False

            for p in selected:
                if running_count > SAFE_MAX - p["numel"]:
                    overflow = True
                    break
                running_count += p["numel"]

            if overflow:
                reasons.append("INVALID_PARAMETER")
                peft_pass = False
                trainable_params = []
                trainable_count = 0
            else:
                trainable_params = [p["name"] for p in selected]
                trainable_count = running_count

    if payload["inferenceMode"] is not False:
        peft_pass = False

    artifact_files = payload["artifactFiles"]

    if not isinstance(artifact_files, list):
        reasons.append("ADAPTER_FILE_SET")
    else:
        adapter_files = sorted(
            set(
                x for x in artifact_files
                if isinstance(x, str)
            ),
            key=bkey,
        )

        expected_files = {
            "adapter_config.json",
            "adapter_model.safetensors",
        }

        supplied_set = set(
            x for x in artifact_files
            if isinstance(x, str)
        )

        if (
            len(artifact_files) != 2
            or len(supplied_set) != 2
            or supplied_set != expected_files
        ):
            reasons.append("ADAPTER_FILE_SET")

        # Full-model artifacts are independently disallowed.
        full_model_names = {
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
            "model.safetensors",
            "model.safetensors.index.json",
            "model.bin",
            "model.bin.index.json",
        }

        if any(
            isinstance(x, str) and x in full_model_names
            for x in artifact_files
        ):
            reasons.append("FULL_MODEL_ARTIFACT")

    if not isinstance(payload["checkpoint"], dict):
        reasons.append("INCOMPLETE_CHECKPOINT")
    else:
        required_checkpoint = {
            "model",
            "optimizer",
            "scheduler",
            "step",
            "rng",
            "dataPosition",
        }

        if not required_checkpoint.issubset(payload["checkpoint"].keys()):
            reasons.append("INCOMPLETE_CHECKPOINT")
        else:
            checkpoint_complete = True

    base_ok = (
        isinstance(payload["baseRevision"], str)
        and SHA40.fullmatch(payload["baseRevision"]) is not None
    )

    if not base_ok:
        reasons.append("MUTABLE_BASE_REVISION")
        lineage_pass = False

    digest_ok = all(
        isinstance(payload[k], str)
        and SHA64.fullmatch(payload[k]) is not None
        for k in ("datasetDigest", "codeDigest", "configDigest")
    )

    expected = payload["expectedDigests"]

    if not isinstance(expected, dict):
        digest_ok = False
    else:
        # Expected digests, when supplied, must bind exactly.
        for k in ("datasetDigest", "codeDigest", "configDigest"):
            if k in expected and expected[k] != payload[k]:
                digest_ok = False

    if not digest_ok:
        reasons.append("LINEAGE_MISMATCH")
        lineage_pass = False

    batch_values = (
        payload["microBatch"],
        payload["gradientAccumulation"],
        payload["replicas"],
        payload["expectedEffectiveBatch"],
    )

    if not all(positive_safe_int(x) for x in batch_values):
        reasons.append("EFFECTIVE_BATCH_MISMATCH")
    else:
        try:
            actual_batch = (
                payload["microBatch"]
                * payload["gradientAccumulation"]
                * payload["replicas"]
            )
        except Exception:
            actual_batch = -1

        if (
            actual_batch != payload["expectedEffectiveBatch"]
            or actual_batch > SAFE_MAX
        ):
            reasons.append("EFFECTIVE_BATCH_MISMATCH")

    train_ids = payload["trainRowIds"]
    eval_ids = payload["evalRowIds"]

    ids_valid = (
        unique_strings(train_ids)
        and unique_strings(eval_ids)
    )

    if not ids_valid:
        reasons.append("EVAL_LEAKAGE")
        eval_isolated = False
    elif set(train_ids) & set(eval_ids):
        reasons.append("EVAL_LEAKAGE")
        eval_isolated = False

    if payload["dropoutActiveDuringEval"] is not False:
        reasons.append("EVAL_DROPOUT_ACTIVE")
        eval_isolated = False
        evaluation_deterministic = False

    u = payload["uninterruptedWeights"]
    r = payload["resumedWeights"]
    tolerance = payload["resumeTolerance"]

    if (
        not isinstance(u, list)
        or not isinstance(r, list)
        or len(u) == 0
        or len(r) == 0
        or len(u) != len(r)
        or not finite(tolerance)
        or tolerance < 0
        or any(not finite(x) for x in u)
        or any(not finite(x) for x in r)
    ):
        reasons.append("RESUME_DIVERGENCE")
        resume_pass = False
    else:
        if any(abs(a - b) > tolerance for a, b in zip(u, r)):
            reasons.append("RESUME_DIVERGENCE")
            resume_pass = False

    # Sort and deduplicate all reason codes by UTF-8 bytes.
    reasons = sorted_codes(reasons)

    return {
        "labels": labels,
        "templatePass": template_pass,
        "trainableParams": trainable_params,
        "trainableCount": trainable_count,
        "peftConfigPass": peft_pass,
        "adapterFiles": adapter_files,
        "checkpointComplete": checkpoint_complete,
        "lineagePass": lineage_pass,
        "evalIsolated": eval_isolated,
        "evaluationDeterministic": evaluation_deterministic,
        "resumePass": resume_pass,
        "reasonCodes": reasons,
    }


@app.post("/adapt")
async def adapt(request: Request):
    try:
        payload = strict_json(await request.body())
    except Exception:
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400,
        )

    if not isinstance(payload, dict):
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400,
        )

    operation = payload.get("operation")

    if operation == "choose":
        return JSONResponse(choose(payload))

    if operation == "repair":
        return JSONResponse(repair(payload))

    return JSONResponse(
        {"error": "INVALID_INPUT"},
        status_code=400,
    )
