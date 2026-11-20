import os
import asyncio
import concurrent.futures
from dataclasses import dataclass
from typing import Optional
import re
from unittest import result
import pandas as pd
from tqdm import tqdm
from botocore.exceptions import ClientError


@dataclass
class ModelConfig:
    model_id: str
    max_tokens: int = 5
    max_concurrent: int = 10

def _invoke_sync(client, model_id, text, system_msg, max_tokens, temperature):

    prefix = text
    suffix = ""

    if "Which response is better" in text:
        parts = text.split("Which response is better")

        if len(parts) == 2:
            prefix = parts[0] + "Which response is better"
            suffix = parts[1].strip()

    if "anthropic" in model_id:

        system_payload = [
            {"text": system_msg},
            {"cachePoint": {"type": "default"}},
        ]

        messages = [
            {
                "role": "user",
                "content": [{"text": f"{prefix}\n\n{suffix}"}],
            }
        ]

    else:

        system_payload = [{"text": system_msg}]

        messages = [
            {
                "role": "user",
                "content": [{"text": text}],
            }
        ]

    return client.converse(
        modelId=model_id,
        system=system_payload,
        messages=messages,
        inferenceConfig={
            "maxTokens": max_tokens,
            "temperature": temperature,
        },
    )

def extract_final_answer(text: str):
    match = re.search(
        r"so\s+the\s+answer\s+is\s*(yes|no)",
        text,
        re.IGNORECASE,
    )
    return match.group(1).lower() if match else None

async def get_llm_response(
    text,
    client,
    model_id,
    valid_responses,
    executor,
    system_msg="",
    retries=3,
    temperature=0.0,
    max_tokens=5,
    use_in_scan=True,
    save_raw=False,
):
    loop = asyncio.get_running_loop()

    for i in range(retries):
        try:
            response = await loop.run_in_executor(
                executor,
                _invoke_sync,
                client,
                model_id,
                text,
                system_msg,
                max_tokens,
                temperature,
            )

            content = response["output"]["message"]["content"]

            result = next(
                (block["text"].strip() for block in content if "text" in block),
                None,
            )

            if result is None:
                result = next(
                    (
                        block["reasoningContent"]["reasoningText"]["text"].strip()
                        for block in content
                        if "reasoningContent" in block
                    ),
                    None,
                )            
            if result is None:
                print(f"[WARN] Empty response, retrying ({i+1}/{retries})")
                continue

            if save_raw:
                return result
            extracted = extract_final_answer(result)
            if extracted:
                return extracted
            if result in valid_responses:
                return result

            first_word = result.strip().split()[0].upper()

            if first_word in [v.upper() for v in valid_responses]:
                return first_word

            if use_in_scan:
                for v in valid_responses:
                    if v.upper() in result.upper():
                        return v

            print(f"[WARN] Invalid response '{result}', retrying")

        except Exception as e:
            if i == retries - 1:
                print(f"[ERROR] {e}. No retries left.")
                break

            wait_time = (2 ** (i + 1)) * 5

            label = "[THROTTLE]" if isinstance(e, ClientError) else "[ERROR]"

            print(f"{label} {e}. Retrying in {wait_time}s")

            await asyncio.sleep(wait_time)

    return None

async def _run_inference_async(
    df,
    model_id,
    client,
    valid_responses,
    max_concurrent,
    system_msg="",
    retries=3,
    temperature=0.0,
    max_tokens=5,
    checkpoint_path=None,
    checkpoint_every=50,
    use_in_scan=True,
    save_raw=False,
):

    prompt_cols = [col for col in df.columns if col.startswith("prompt_")]

    total_calls = len(df) * len(prompt_cols)

    print(f"[INFO] Starting inference: {len(df)} rows x {len(prompt_cols)} prompts = {total_calls} total calls")
    print(f"[INFO] Concurrency limit: {max_concurrent}")

    semaphore = asyncio.Semaphore(max_concurrent)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent)

    tasks = []

    skipped = 0

    for index, row in df.iterrows():
        for prompt_col in prompt_cols:

            answer_col = prompt_col.replace("prompt_", "answer_", 1)

            if answer_col not in df.columns:
                df[answer_col] = None

            if answer_col in df.columns and pd.notna(df.at[index, answer_col]):
                skipped += 1
                continue

            tasks.append((index, prompt_col, answer_col, row[prompt_col]))

    print(f"[INFO] Skipping {skipped} existing answers")

    async def run_single(index, prompt_col, answer_col, text):
        async with semaphore:
            result = await get_llm_response(
                text,
                client,
                model_id,
                valid_responses,
                executor,
                system_msg,
                retries,
                temperature,
                max_tokens,
                use_in_scan,
                save_raw,
            )
            return index, answer_col, result

    pbar = tqdm(total=len(tasks), desc="Calls")

    checkpoint_counter = 0

    for coro in asyncio.as_completed(
        [run_single(i, pc, ac, txt) for i, pc, ac, txt in tasks]
    ):
        index, answer_col, result = await coro

        if save_raw and result is not None:
            match = re.search(r"my final answer is:\s*(yes|no)", result, re.IGNORECASE)
            if match:
                result = match.group(1).lower()
            else:
                text_stripped = result.strip().lower()
                if text_stripped in ["yes", "no", "yes.", "no."]:
                    result = text_stripped.rstrip(".")

        df.at[index, answer_col] = result

        pbar.update(1)

        checkpoint_counter += 1

        if checkpoint_path and checkpoint_counter % checkpoint_every == 0:
            df.to_csv(checkpoint_path, index=False)
            print(f"[CHECKPOINT] Saved {checkpoint_counter} calls")

    pbar.close()

    executor.shutdown(wait=False)

    return df

def run_inference(
    df,
    model_id,
    client,
    valid_responses,
    system_msg="",
    retries=3,
    temperature=0.0,
    max_tokens=5,
    checkpoint_path=None,
    checkpoint_every=50,
    use_in_scan=True,
    max_concurrent=10,
    save_raw=False,
):

    return asyncio.run(
        _run_inference_async(
            df=df,
            model_id=model_id,
            client=client,
            valid_responses=valid_responses,
            max_concurrent=max_concurrent,
            system_msg=system_msg,
            retries=retries,
            temperature=temperature,
            max_tokens=max_tokens,
            checkpoint_path=checkpoint_path,
            checkpoint_every=checkpoint_every,
            use_in_scan=use_in_scan,
            save_raw=save_raw
        )
    )

def run_all_models(
    df,
    models,
    client,
    valid_responses,
    system_msg="",
    system_msg_overrides=None,
    retries=3,
    temperature=0.0,
    max_tokens=5,
    save_dir=None,
    use_in_scan=True,
    save_raw=False,
):

    if system_msg_overrides is None:
        system_msg_overrides = {}

    results = {}

    for model_name, config in models.items():

        model_id = config.model_id
        model_max_tokens = max_tokens if max_tokens is not None else config.max_tokens
        model_max_concurrent = config.max_concurrent

        model_system_msg = system_msg_overrides.get(model_name, system_msg)

        print(f"\n[MODEL] Running {model_name} ({model_id})")

        df_model = df.copy()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

            partial_path = f"{save_dir}/{model_name}_results.csv"

            if os.path.exists(partial_path):
                print(f"[MODEL] Resuming from checkpoint")
                df_model = pd.read_csv(partial_path)
                answer_cols = [c for c in df_model.columns if c.startswith("answer_")]
                df_model[answer_cols] = df_model[answer_cols].astype(object)

        df_model = run_inference(
            df_model,
            model_id,
            client,
            valid_responses,
            system_msg=model_system_msg,
            retries=retries,
            temperature=temperature,
            max_tokens=model_max_tokens,
            checkpoint_path=f"{save_dir}/{model_name}_results.csv" if save_dir else None,
            use_in_scan=use_in_scan,
            max_concurrent=model_max_concurrent,
            save_raw=save_raw,
        )

        df_model["model_name"] = model_name

        results[model_name] = df_model

        if save_dir:
            df_model.to_csv(f"{save_dir}/{model_name}_results.csv", index=False)

        print(f"[MODEL] Finished {model_name}")

    print("\n[INFO] All models complete.")

    return results
    