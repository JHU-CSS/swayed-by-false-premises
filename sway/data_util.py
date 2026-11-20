import pandas as pd
from .presuppositions import AITA_PRESUPPOSITIONS,DQA_PRESUPPOSITIONS, LFQA_PRESUPPOSITIONS, Presupposition

def load_and_sample(path, n=10, filetype="csv", dropna=False):
    if filetype == "csv":
        df = pd.read_csv(path)
    elif filetype == "json":
        df = pd.read_json(path)
    elif filetype == "jsonl":
        df = pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unsupported filetype: {filetype}")

    if dropna:
        df = df.dropna()

    return df.head(n)

def build_prompt_variants(df, base_col, ps):
    df = df.copy()
    for p in ps:
        col_name = f"{p.name}_{p.commitment.name.lower()}_{p.description.lower().replace(' ', '_')}"
        df[f"prompt_{col_name}"] = df[base_col] + " " + p.suffix
        df[f"answer_{col_name}"] = None
    return df

def load_aita(yta_path,nta_path, n):
    if n % 2 != 0:
        raise ValueError(f"n must be even for a 50/50 split, got {n}.")

    half = n // 2

    yta = pd.read_csv(yta_path)[["prompt", "ytanta"]].dropna()
    yta = yta.rename(columns={"ytanta": "outcome"}).sample(n=half, random_state=42)
    nta = pd.read_csv(nta_path)[["original_post"]].dropna()
    nta = nta.rename(columns={"original_post": "prompt"}).sample(n=half, random_state=42)
    nta["outcome"] = "NTA"

    df = pd.concat([yta, nta], ignore_index=True)

    return build_prompt_variants(df, base_col="prompt", ps=AITA_PRESUPPOSITIONS)

def load_lfqa(path, n=None, max_words=None):
    df = pd.read_json(path)

    pairs = []

    paired_columns = [
        ("student_answer_a_en", "student_answer_b_en"),
        ("model_answer_a_en", "model_answer_b_en"),
    ]

    for a_col, b_col in paired_columns:
        valid = df[df[a_col].notna() & df[b_col].notna()].copy()

        temp = valid[["question", a_col, b_col]].rename(columns={
            a_col: "response_a",
            b_col: "response_b"
        })

        pairs.append(temp)

    df = pd.concat(pairs, ignore_index=True)

    if max_words is not None:
        def _wc(text):
            return len(str(text).split())
        combined_wc = (
            df["question"].apply(_wc)
            + df["response_a"].apply(_wc)
            + df["response_b"].apply(_wc)
        )
        df = df[combined_wc <= max_words].reset_index(drop=True)

    if n is not None:
        df = df.head(n)

    df["base_prompt"] = df.apply(
        lambda row: (
            f"Question: {row['question']}\n\n"
            f"Response A: {row['response_a']}\n\n"
            f"Response B: {row['response_b']}\n\n"
            f"{LFQA_PRESUPPOSITIONS.shared_prefix} "
            f"{LFQA_PRESUPPOSITIONS.instruction}"
        ),
        axis=1,
    )

    df = df.copy()

    for p in LFQA_PRESUPPOSITIONS.presuppositions:
        col_name = f"{p.name}_{p.commitment.name.lower()}_{p.description.lower().replace(' ', '_')}"

        df[f"prompt_{col_name}"] = df["base_prompt"] + " " + p.suffix

        df[f"answer_{col_name}"] = None

    return df

def load_dqa(path, n):
    path_str = str(path)
    if path_str.endswith(".jsonl"):
        df = pd.read_json(path, lines=True)
    else:
        df = pd.read_json(path)

    df = df[["question"]].dropna().copy()

    allowed_starters = ("is ", "should ", "do ", "does ")
    df = df[df["question"].str.strip().str.lower().str.startswith(allowed_starters)].reset_index(drop=True)

    if n is not None:
        df = df.head(n)

    return build_prompt_variants(df, base_col="question", ps=DQA_PRESUPPOSITIONS)