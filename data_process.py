from transformers import AutoTokenizer
import tree_sitter_java
from tree_sitter import Parser, Language
from datasets import load_dataset



# Initialize tools
print("Loading CodeBERT tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
print("Initializing Java parser...")
JAVA_LANGUAGE = Language(tree_sitter_java.language())
parser = Parser()
parser.language = JAVA_LANGUAGE



# Data flow extraction function
def extract_data_flow(code):
    """
    Extract variable list and data flow edges from Java code.
    """
    variables = []
    edges = []

    tree = parser.parse(bytes(code, "utf8"))
    root_node = tree.root_node

    def traverse_node(node):
        if node.type == 'local_variable_declaration':
            declarators = []
            for child in node.children:
                if child.type == 'variable_declarator':
                    declarators.append(child)
                elif child.type == 'variable_declarator_list':
                    declarators.extend([c for c in child.children if c.type == 'variable_declarator'])

            for declarator in declarators:
                var_name_node = declarator.child_by_field_name('name')
                if var_name_node is None:
                    continue
                var_name = code[var_name_node.start_byte:var_name_node.end_byte]
                variables.append(var_name)

                def collect_right_identifiers(n, left_var_name):
                    if n.type == 'identifier':
                        ident = code[n.start_byte:n.end_byte]
                        if ident != left_var_name:
                            variables.append(ident)
                            edges.append((ident, left_var_name))
                    for child in n.children:
                        collect_right_identifiers(child, left_var_name)

                for child in declarator.children:
                    if child is var_name_node:
                        continue
                    collect_right_identifiers(child, var_name)

        for child in node.children:
            traverse_node(child)

    traverse_node(root_node)
    # Deduplicate
    variables = list(set(variables))
    return variables, edges



#  Download and sample BigCloneBench dataset
print("Downloading BigCloneBench dataset...")
dataset = load_dataset("code_x_glue_cc_clone_detection_big_clone_bench")

# Sample 50% of training set to fit time and GPU memory (4060Ti)
train_dataset = dataset["train"].train_test_split(test_size=0.5, seed=42)["train"]
val_dataset = dataset["validation"]
test_dataset = dataset["test"]
print(f"Sampled training set size: {len(train_dataset)}, validation set: {len(val_dataset)}, test set: {len(test_dataset)}")


# Preprocessing: input construction following paper
MAX_SEQ_LEN = 448  # suitable for 4060Ti
MAX_NODE_NUM = 96


def process_function(examples):
    code1 = examples["func1"]
    code2 = examples["func2"]
    label = examples["label"]

    # Extract data flow for both code snippets
    vars1, edges1 = extract_data_flow(code1)
    vars2, edges2 = extract_data_flow(code2)

    # Truncate variables and edges to max length
    vars1 = vars1[:MAX_NODE_NUM]
    vars2 = vars2[:MAX_NODE_NUM]
    edges1 = [(u, v) for u, v in edges1 if u in vars1 and v in vars1]
    edges2 = [(u, v) for u, v in edges2 if u in vars2 and v in vars2]

    # Concatenate input: code tokens + [SEP] + variable tokens
    # First code snippet
    tokens_code1 = tokenizer.tokenize(code1)
    tokens_vars1 = tokenizer.tokenize(" ".join(vars1))
    input_tokens1 = [tokenizer.cls_token] + tokens_code1 + [tokenizer.sep_token] + tokens_vars1 + [tokenizer.sep_token]

    # Second code snippet
    tokens_code2 = tokenizer.tokenize(code2)
    tokens_vars2 = tokenizer.tokenize(" ".join(vars2))
    input_tokens2 = [tokenizer.cls_token] + tokens_code2 + [tokenizer.sep_token] + tokens_vars2 + [tokenizer.sep_token]

    # Truncate to max sequence length
    input_tokens1 = input_tokens1[:MAX_SEQ_LEN]
    input_tokens2 = input_tokens2[:MAX_SEQ_LEN]

    # Convert to token ids
    input_ids1 = tokenizer.convert_tokens_to_ids(input_tokens1)
    input_ids2 = tokenizer.convert_tokens_to_ids(input_tokens2)

    return {
        "input_ids1": input_ids1,
        "input_ids2": input_ids2,
        "vars1": vars1,  # list of variable names
        "vars2": vars2,
        "edges1": edges1,
        "edges2": edges2,
        "label": label
    }



#  Process datasets
print("Starting preprocessing of training set...")
processed_train = train_dataset.map(process_function, remove_columns=train_dataset.column_names)

print("Starting preprocessing of validation set...")
processed_val = val_dataset.map(process_function, remove_columns=val_dataset.column_names)

print("Starting preprocessing of test set...")
processed_test = test_dataset.map(process_function, remove_columns=test_dataset.column_names)

# Save preprocessed datasets to disk
processed_train.save_to_disk("./processed_train")
processed_val.save_to_disk("./processed_val")
processed_test.save_to_disk("./processed_test")
print("Preprocessing completed! All datasets saved locally.")