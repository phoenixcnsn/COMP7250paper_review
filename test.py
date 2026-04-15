from transformers import AutoTokenizer, AutoModel
import tree_sitter_java
from tree_sitter import Parser, Language

print("Loading CodeBERT base model...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
print("CodeBERT loaded!")

print("Initializing Java parser...")
JAVA_LANGUAGE = Language(tree_sitter_java.language())
parser = Parser()
parser.language = JAVA_LANGUAGE
print("Java parser initialized!")

test_code = """
public class Test {
    public int add(int a, int b) {
        int x = a + b;
        return x;
    }
}
"""

tree = parser.parse(bytes(test_code, "utf8"))
root_node = tree.root_node

variables = []
edges = []

def traverse_node(node):
    if node.type == 'local_variable_declaration':
        declarators = []
        for child in node.children:
            if child.type == 'variable_declarator':
                declarators.append(child)
            elif child.type == 'variable_declarator_list':
                declarators.extend([c for c in child.children if c.type == 'variable_declarator'])

        for declarator in declarators:
            # Get the left-hand variable name
            var_name_node = declarator.child_by_field_name('name')
            if var_name_node is None:
                continue
            var_name = test_code[var_name_node.start_byte:var_name_node.end_byte]
            variables.append(var_name)

            # Collect all identifiers on the right-hand side (recursively traverse all descendants of declarator, but skip the left-hand variable name itself)
            def collect_right_identifiers(n, left_var_name):
                # If current node is an identifier and its text is not equal to left_var_name (avoid treating left variable as right)
                if n.type == 'identifier':
                    ident = test_code[n.start_byte:n.end_byte]
                    if ident != left_var_name:   # Simple deduplication of left variable itself (though left variable won't appear in right subtree normally)
                        variables.append(ident)
                        edges.append((ident, left_var_name))
                for child in n.children:
                    collect_right_identifiers(child, left_var_name)

            # Recursively collect from all children of declarator, but skip the name node
            for child in declarator.children:
                if child is var_name_node:
                    continue   # Skip the left-hand variable name
                collect_right_identifiers(child, var_name)

    for child in node.children:
        traverse_node(child)

traverse_node(root_node)

variables = list(set(variables))

print("\n=== Data Flow Extraction Test Results ===")
print(f"Extracted variables: {variables}")
print(f"Extracted data flow edges (right variable -> left variable): {edges}")
