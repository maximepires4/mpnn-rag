import ast


class CodeContextExtractor(ast.NodeVisitor):
    def __init__(self):
        self.scopes = []
        self.scope_stack = []

    def _get_full_scope_name(self):
        return ".".join(self.scope_stack)

    def visit_ClassDef(self, node):
        self.scope_stack.append(node.name)
        full_name = self._get_full_scope_name()
        self.scopes.append((node.lineno, node.end_lineno, full_name, "class"))
        self.generic_visit(node)
        self.scope_stack.pop()

    def _visit_function(self, node):
        self.scope_stack.append(node.name)
        full_name = self._get_full_scope_name()
        self.scopes.append((node.lineno, node.end_lineno, full_name, "function"))
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_FunctionDef(self, node):
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node):
        self._visit_function(node)


def get_code_context(source_code):
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []

    extractor = CodeContextExtractor()
    extractor.visit(tree)
    return extractor.scopes


def enrich_python_metadata(original_doc, chunks):
    """
    Ajoute les métadonnées (classe, fonction) aux chunks en les mappant
    vers le document original via les numéros de lignes.
    """
    source_code = original_doc.page_content
    scopes = get_code_context(source_code)

    # S'il n'y a pas de code parsable, on retourne les chunks tels quels
    if not scopes:
        return chunks

    current_idx = 0

    for chunk in chunks:
        start_char_idx = source_code.find(chunk.page_content, current_idx)

        if start_char_idx == -1:
            start_char_idx = source_code.find(chunk.page_content)

        if start_char_idx != -1:
            current_idx = start_char_idx + len(chunk.page_content)

            start_line = source_code.count("\n", 0, start_char_idx) + 1
            end_line = start_line + chunk.page_content.count("\n")

            best_scope = None

            for s_start, s_end, s_name, s_type in scopes:
                if s_start <= start_line and s_end >= end_line:
                    best_scope = (s_name, s_type)

            if best_scope:
                chunk.metadata["context"] = best_scope[0]
                chunk.metadata["context_type"] = best_scope[1]

            chunk.metadata["start_line"] = start_line
            chunk.metadata["end_line"] = end_line

    return chunks
