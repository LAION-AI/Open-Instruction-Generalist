"""Text2SQL dataset.

Collection of few shot Spider datasets with the conversational CoSQL and Sparc datasets.
"""
import copy
import json
import multiprocessing
import os
import random
import re
import zipfile
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Dict

import gdown
import numpy as np
import sqlglot
import torch
from sentence_transformers import SentenceTransformer, util
from sqlglot import parse_one
from tqdm.auto import tqdm

DATASETS_TO_URL = {
    "spider": "https://drive.google.com/uc?export=download&id=1_AckYkinAnhqmRQtGsQgUKAnTHxxX5J0",
    "sparc": "https://drive.google.com/uc?export=download&id=1Uu7NMHTR1tdQw1t7bAuM7OPU4LElVKfg",
    "cosql": "https://drive.google.com/uc?export=download&id=1Y3ydpFiQQ3FC0bzdfy3groV95O_f1nXF",
}


class EmbeddingSimilaritySelector:
    """Sentence Transformer embedding similary selection."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_seq_length: int = 512,
    ) -> None:
        """Initialize embedding similarity selector model.

        Use sentence_transformers to embed demonstrations and queries for cosine
        similarity scoring.

        Args:
            model_name: name of sentence_transformers model
            max_seq_length: max sequence length for sentence_transformers model
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.similarity_model = SentenceTransformer(model_name, device=self.device)
        self.similarity_model.max_seq_length = max_seq_length
        self._cache: dict = {}

    def embed(self, text: str) -> torch.Tensor:
        """Embed text."""
        hsh = hash(text)
        if hsh in self._cache:
            return self._cache[hsh]
        embeds = self.similarity_model.encode(
            text, convert_to_tensor=True, device=self.device
        )
        self._cache[hsh] = embeds
        return embeds

    def _fill_cache(self, inputs: list[str]) -> None:
        """Fill cache with embeddings."""
        embeds = self.similarity_model.encode(
            inputs, convert_to_tensor=True, device=self.device
        )
        for i, hsh in enumerate(map(hash, inputs)):
            self._cache[hsh] = embeds[i]

    def _select_indices(
        self, query: str, demonstrations: list[tuple[str, str]]
    ) -> np.ndarray:
        """Select k most similar texts."""
        query_embed = torch.stack([self.embed(query)])
        # Only embed input
        texts_embed = torch.stack([self.embed(x[0]) for x in demonstrations])
        similarities = util.pytorch_cos_sim(query_embed, texts_embed)
        similarities = similarities.cpu().detach().numpy()[0]
        indices = np.argsort(similarities)[::-1]
        return indices


class PromptFormatter:
    """Formatter for assistant."""

    table_sep: str = "\n"

    @classmethod
    def format_table(cls, table: Dict) -> str:
        """Get table format."""
        table_fmt = []
        table_name = table["name"]
        for col in table["columns"] or []:
            table_fmt.append(f"{col['name']}")
        if table_fmt:
            all_cols = ", ".join(table_fmt)
            create_tbl = f"{table_name} ({all_cols})"
        else:
            create_tbl = f"{table_name}"
        return create_tbl

    @classmethod
    def format_prompt(
        cls,
        instruction: str,
        table_text: str,
        is_demonstration: bool,
    ) -> str:
        """Get prompt format."""
        schema_str = f"""Schema:\n{table_text}"""
        if not instruction.startswith("User:"):
            pmt = f"""User: {instruction}\n{schema_str}\nAssistant: """  # noqa: E501
        else:
            pmt = f"""{instruction}\n{schema_str}\nAssistant: """
        if not is_demonstration:
            pmt += "SELECT"
        return pmt

    @classmethod
    def format_model_output(cls, output_sql: str) -> str:
        """Format model output.

        Our prompt ends with select so we need to add it back.
        """
        if not output_sql.lower().startswith("select"):
            output_sql = "SELECT " + output_sql
        return output_sql

    @classmethod
    def format_gold_output(cls, output_sql: str) -> str:
        """Format gold output for demonstration.

        Our prompt ends with SELECT so we need to remove it.
        """
        if not output_sql.strip().endswith(";"):
            output_sql += ";"
        return output_sql


class Text2SQLDataset(ABC):
    """Text2SQL dataset."""

    def __init__(
        self,
        name: str,
        train_data_file: str,
        val_data_file: str,
        schema_file: str,
        **kwargs: Any,
    ) -> None:
        """Initialize."""
        self.name = name
        self.train_data_file = train_data_file
        self.val_data_file = val_data_file
        self.schema_file = schema_file

    @abstractmethod
    def load_data(
        self, schema: dict[str, dict[str, dict]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Load data."""
        raise NotImplementedError

    @abstractmethod
    def load_schema(self) -> dict[str, dict[str, dict]]:
        """Load schema."""
        raise NotImplementedError

    def format_example(
        self,
        example: dict[str, Any],
        schema: dict[str, dict[str, dict]],
        prompt_formatter: PromptFormatter,
    ) -> dict | None:
        """Format example."""
        gold_sql_key = "sql"

        def _is_parseable(sql: str) -> bool:
            try:
                res: sqlglot.expressions.Expression | None = parse_one(sql)
                return res is not None
            except Exception:
                return False

        def _format_example(
            ex: dict[str, Any],
            is_demonstration: bool,
        ) -> tuple[str, str] | None:
            # Take the sql part only if pretask is added
            if not _is_parseable(ex[gold_sql_key].split("|")[-1]):
                print("BAD:::", ex[gold_sql_key].split("|")[-1])
                return None

            db_id = ex.get("db_id", "database")
            db_schema = schema[db_id]
            tables_to_add = list(db_schema.keys())

            # Shuffle to make sure model doesn't learn about order
            # random.shuffle(tables_to_add)
            question = clean_str(ex["question"]).strip("'").strip('"')
            table_text = prompt_formatter.table_sep.join(
                [prompt_formatter.format_table(db_schema[t]) for t in tables_to_add]
            )
            input_str = prompt_formatter.format_prompt(
                question,
                table_text,
                is_demonstration=is_demonstration,
            )
            output_str = prompt_formatter.format_gold_output(ex[gold_sql_key])
            return input_str, output_str

        demonstration_strs = []
        for _, demo in enumerate(example.get("demonstrations", [])):
            formatted = _format_example(demo, True)
            if formatted:
                demonstration_strs.append(formatted[0] + formatted[1])
        # Use 0 example id to get the schema and True to not start with SELECT
        result = _format_example(example, False)
        if not result:
            return None
        input_str, output_str = result
        input_str = "\n\n".join(demonstration_strs + [input_str])
        data_ex = dict(
            input=input_str,
            output=output_str,
            db_id=example.get("db_id", "database"),
            dataset=self.name,
            question=example["question"],
            sql_with_prefix=example[gold_sql_key],
            sql=example[gold_sql_key],
        )
        return data_ex


class SpiderText2SQL(Text2SQLDataset):
    """Spider text2sql dataset adapted from Huggingface/Picard."""

    DEMONSTRATION_SELECTOR = EmbeddingSimilaritySelector()

    def _select_demonstrations(
        self,
        example: dict[str, Any],
        all_data: list[dict],
        db_id_index: dict[str, list[int]],
    ) -> list[dict]:
        """Select demonstrations."""
        num_demos = 5
        query = example["question"]
        db_id_to_filter = example["db_id"]
        gold_sql = example.get("sql", None)

        # Filter away any disallowed demonstrations
        as_text_demos: list[tuple[str, str]] = []
        as_dict_demos: list[dict] = []
        indexes = db_id_index[db_id_to_filter]
        for i in indexes:
            ex = all_data[i]
            # This makes sure we skip exact matches in case demo data == test data
            if re.sub(r"\s+", " ", ex["question"].lower()) == re.sub(
                r"\s+", " ", query.lower()
            ):
                continue
            if gold_sql:
                # Make sure to remove any prefix task parts from the SQL
                as_text_sql = parse_one(ex["sql"].split("|")[-1].strip()).sql().lower()
                gold_sql_sql = parse_one(gold_sql.split("|")[-1].strip()).sql().lower()
                if gold_sql_sql == as_text_sql:
                    continue
            as_text_demos.append((ex["question"], ex["sql"]))
            as_dict_demos.append(ex)
        # Select demonstrations
        final_indices = []
        if as_text_demos:
            indices = self.DEMONSTRATION_SELECTOR._select_indices(
                query=query,
                demonstrations=as_text_demos,  # type: ignore
            )
            final_indices = indices.tolist()

        selected_demos = []
        # Make sure we don't take duplicate sqls
        seen_sql = set()
        for idx in final_indices:
            if len(selected_demos) >= num_demos:
                break
            ex = as_dict_demos[idx]
            if ex["sql"] in seen_sql:
                continue
            seen_sql.add(ex["sql"])
            selected_demos.append(ex)
        selected_demos = selected_demos[::-1]
        return selected_demos

    def load_data(self) -> dict[str, list[dict[str, Any]]]:
        """Load data."""
        splits: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
        all_data_for_demos: dict[str, list[dict[str, Any]]] = {
            "train": [],
            "dev": [],
        }
        for split in splits:
            if split in "dev":
                to_read_file = self.val_data_file
            elif split == "train":
                to_read_file = self.train_data_file
            else:
                raise ValueError(f"Unknown split {split}")
            print(f"Loading {split} data")
            data_file = Path(to_read_file)
            try:
                data = json.load(data_file.open())
            except json.decoder.JSONDecodeError:
                data = [json.loads(line) for line in data_file.open()]
            for raw_ex in tqdm(
                data, desc=f"Loading {split} data from {data_file.name}"
            ):
                query = raw_ex["query"]
                db_id = raw_ex["db_id"]
                target = case_sql(query)
                ex = {
                    "question": raw_ex["question"],
                    "db_id": db_id,
                    "sql": target,
                    "raw_sql": raw_ex["query"],
                    "parsed_sql": raw_ex["sql"],
                }
                splits[split].append(ex)
                all_data_for_demos[split].append(copy.deepcopy(ex))

        # Now go back and add demonstrations
        for split in splits:
            # Build database to index
            db_id_index = defaultdict(list)
            seen_keys = set()
            for i, ex in enumerate(all_data_for_demos[split]):
                db_id = str(ex["db_id"])
                sql = ex["sql"]
                if not sql:
                    raise ValueError("Demonstration data must have a query field.")
                sql = re.sub(r"\s+", " ", str(sql)).strip()
                question = str(ex["question"])
                key = (question, sql, db_id)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                db_id_index[db_id].append(i)
            # Do it in parallel
            pool = multiprocessing.Pool(processes=10)
            _select_demo = partial(
                self._select_demonstrations,
                all_data=all_data_for_demos[split],
                db_id_index=db_id_index,
            )
            all_demonstration_sets = list(
                pool.imap(
                    _select_demo,
                    tqdm(splits[split], desc=f"Adding {split} demonstrations"),
                    chunksize=100,
                )
            )
            pool.close()
            pool.join()
            for i, ex in enumerate(splits[split]):
                demonstrations = all_demonstration_sets[i]
                ex["demonstrations"] = demonstrations
        return splits

    def load_schema(self) -> dict[str, dict[str, dict]]:
        """Load schema for each table in the database."""
        schema_dct = read_tables_json(
            self.schema_file,
            lowercase=True,
        )
        return schema_dct


class SparcText2SQL(SpiderText2SQL):
    """Sparc text2sql dataset."""

    def load_data(self) -> dict[str, list[dict[str, Any]]]:
        """Load data."""
        splits: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
        for split in splits:
            if split in "dev":
                to_read_file = self.val_data_file
            elif split == "train":
                to_read_file = self.train_data_file
            else:
                raise ValueError(f"Unknown split {split}")
            data_file = Path(to_read_file)
            data = json.load(data_file.open())
            for raw_ex in tqdm(data, desc=f"Loading {split} data"):
                db_id = raw_ex["database_id"]
                demonstrations = []
                results = []
                for query_dct in raw_ex["interaction"]:
                    query = re.sub(r"\s+", " ", query_dct["query"].strip())
                    if not query.endswith(";"):
                        query += ";"
                    if query.endswith("DESC 1;"):
                        query = query.replace("DESC 1", "DESC LIMIT 1")
                    if "> =" in query:
                        query = query.replace("> =", ">=")
                    if "< =" in query:
                        query = query.replace("< =", "<=")
                    if "( * )" in query:
                        query = query.replace("( * )", "(*)")
                    question = query_dct["utterance"]
                    target = case_sql(query)
                    sql = {
                        "question": question,
                        "demonstrations": demonstrations,
                        "db_id": db_id,
                        "sql": target,
                        "raw_sql": query_dct["query"],
                        "parsed_sql": query_dct["sql"],
                    }
                    results.append(sql)
                    demonstrations.append(copy.deepcopy(sql))
                for ex in results:
                    if ex:
                        splits[split].append(ex)
        return splits


def read_tables_json(
    schema_file: str,
    lowercase: bool = False,
) -> dict[str, dict[str, dict]]:
    """Read tables json."""
    data = json.load(open(schema_file))
    db_to_tables = {}
    for db in data:
        db_name = db["db_id"]
        db["table_names_original"] = db["table_names_original"]
        table_names = db["table_names_original"]
        db["column_names_original"] = [
            [x[0], x[1]] for x in db["column_names_original"]
        ]
        db["column_types"] = db["column_types"]
        if lowercase:
            table_names = [tn.lower() for tn in table_names]
        tables = defaultdict(list)
        for _, ((ti, col_name), col_type) in enumerate(
            zip(db["column_names_original"], db["column_types"])
        ):
            if ti == -1:
                continue
            if lowercase:
                col_name = col_name.lower()
                col_type = col_type.lower()
            tables[table_names[ti]].append(dict(name=col_name, dtype=col_type))
        db_to_tables[db_name] = {
            table_name: dict(
                name=table_name,
                columns=tables[table_name],
            )
            for table_name in tables
        }
    return db_to_tables


def case_sql(query: str) -> str:
    """Case sql query."""
    try:
        cased_sql = parse_one(query).sql()  # type: ignore
        # SQLGlot makes NOT <col> IN. We want <col> NOT IN for Spider
        cased_sql = re.sub(r"NOT\s+([^\s]+)\s+IN", r"\1 NOT IN", cased_sql)
        # Replace <> with !=
        cased_sql = cased_sql.replace("<>", "!=")
        return cased_sql
    except Exception:
        return query


def clean_str(target: str) -> str:
    """Clean string for question."""
    if not target:
        return target

    target = re.sub(r"[^\x00-\x7f]", r" ", target)
    line = re.sub(r"''", r" ", target)
    line = re.sub(r"``", r" ", line)
    line = re.sub(r"\"", r"'", line)
    line = re.sub(r"[\t ]+", " ", line)
    return line.strip()


def process_dataset(
    prompt_formatter: PromptFormatter,
    splits: dict[str, list[dict]],
    bad_parses: dict[str, int],
    total: dict[str, int],
    text2sql_dataset: Text2SQLDataset,
) -> None:
    """Process a dataset and add it to the splits."""
    schema = text2sql_dataset.load_schema()
    formatting_func = partial(
        text2sql_dataset.format_example,
        schema=schema,
        prompt_formatter=prompt_formatter,
    )
    for split, split_data in text2sql_dataset.load_data().items():
        print(f"Found {len(split_data)} examples for {split}.")
        for formatted_data in tqdm(
            map(formatting_func, split_data),
            total=len(split_data),
            desc=f"Formatting {split}",
        ):
            total[split] += 1
            if formatted_data:
                splits[split].append(formatted_data)
            else:
                bad_parses[split] += 1
    print(f"Bad parses: {json.dumps(bad_parses, indent=2)}")


def download_datasets(output_dir: str):
    """Download with gdown"""
    if not os.path.exists(os.path.join(output_dir)):
        os.makedirs(os.path.join(output_dir))
    for dataset, gdown_path in DATASETS_TO_URL.items():
        gdown.download(
            gdown_path, os.path.join(output_dir, f"{dataset}.zip"), quiet=False
        )
        # Unzip the file
        with zipfile.ZipFile(
            os.path.join(output_dir, f"{dataset}.zip"), "r"
        ) as zip_ref:
            zip_ref.extractall(os.path.join(output_dir))


def create_text2sql(output, data_dir: str = "data/text2sql"):
    """Create text2sql dataset."""
    print(f"Dowloading datasets to {data_dir}...")
    # Set multiprocessing start method to spawn if not already set
    multiprocessing.set_start_method("spawn", force=True)
    # download_datasets(data_dir)
    prompt_formatter = PromptFormatter()
    data_classes = [
        SpiderText2SQL(
            name="spider",
            train_data_file=os.path.join(data_dir, "spider/train_spider.json"),
            val_data_file=os.path.join(data_dir, "spider/dev.json"),
            schema_file=os.path.join(data_dir, "spider/tables.json"),
        ),
        SparcText2SQL(
            name="sparc",
            train_data_file=os.path.join(data_dir, "sparc/train.json"),
            val_data_file=os.path.join(data_dir, "sparc/dev.json"),
            schema_file=os.path.join(data_dir, "sparc/tables.json"),
        ),
        SparcText2SQL(
            name="cosql",
            train_data_file=os.path.join(
                data_dir, "cosql_dataset/sql_state_tracking/cosql_train.json"
            ),
            val_data_file=os.path.join(
                data_dir, "cosql_dataset/sql_state_tracking/cosql_dev.json"
            ),
            schema_file=os.path.join(data_dir, "cosql_dataset/tables.json"),
        ),
    ]

    splits: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    bad_parses: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    for data_class in data_classes:
        print(f"Loading {data_class.name}")
        process_dataset(
            prompt_formatter=prompt_formatter,
            splits=splits,
            bad_parses=bad_parses,
            total=total,
            text2sql_dataset=data_class,
        )

    print(f"Starting length of train: {len(splits['train'])}")
    # Deduplicate training data
    unq_inps = set()
    new_train = []
    for ex in splits["train"]:
        if ex["input"] not in unq_inps:
            new_train.append(ex)
            unq_inps.add(ex["input"])
    splits["train"] = new_train
    print(f"After dedup length of train: {len(splits['train'])}")

    # Save the data
    random.seed(0)
    random.shuffle(splits["train"])

    # Only save train
    split = "train"
    print(
        f"Found {bad_parses[split]} bad parses out of "
        f"{total[split]} ({100*bad_parses[split]/max(total[split], 1): .2f})."
    )
    print(f"Saving {split} ({len(splits[split])}) " f"data to {output}")
    for formatted_ex in splits[split]:
        output.write(json.dumps(formatted_ex) + "\n")


if __name__ == "__main__":
    output = open("_temp_text2sql.jsonl", "w")
    create_text2sql(output)
