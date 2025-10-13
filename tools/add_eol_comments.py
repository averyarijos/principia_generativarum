"""Utility: add end-of-line comments to code lines in a Python file.

This script preserves leading/trailing whitespace and leaves top-level
and block docstrings unmodified. It appends a short comment to every
non-empty line that is not inside a triple-quoted string.

Usage:
    python tools/add_eol_comments.py \
        /workspaces/projects/PrincipiaGenerativarum/generativity/generative_logic.py \
        /workspaces/projects/PrincipiaGenerativarum/generativity/generative_logic_commented.py
"""
import sys
import io
import re

TRIPLE = ('"""', "'''")

def add_comments(src: str) -> str:
    """Return new source with EOL comments on code lines (not inside docstrings)."""
    out_lines = []
    in_doc = False
    doc_delim = None
    for line in src.splitlines(keepends=False):
        stripped = line.strip()
        # detect start/end of triple-quoted blocks
        if not in_doc:
            for d in TRIPLE:
                if stripped.startswith(d):
                    # enters docstring block; if it also ends on same line, treat as docline
                    if stripped.endswith(d) and len(stripped) >= len(d)*2:
                        # single-line docstring, leave unchanged
                        pass
                    else:
                        in_doc = True
                        doc_delim = d
                    break
        else:
            # currently inside docstring
            if doc_delim and doc_delim in stripped:
                in_doc = False
                doc_delim = None
            out_lines.append(line)
            continue

        # If line is blank or a comment-only line, leave as-is
        if stripped == '' or stripped.startswith('#'):
            out_lines.append(line)
            continue

        # For import lines or code lines, append a concise comment
        # Avoid adding comment inside string literals by checking for quotes occurrence
        # We'll naively append "  # auto: code" which is safe as an EOL comment.
        new_line = line + '  # auto: code'
        out_lines.append(new_line)
    return '\n'.join(out_lines) + '\n'


def main():
    if len(sys.argv) != 3:
        print('Usage: add_eol_comments.py input.py output.py')
        sys.exit(2)
    src_path = sys.argv[1]
    dst_path = sys.argv[2]
    with open(src_path, 'r', encoding='utf-8') as f:
        src = f.read()
    out = add_comments(src)
    with open(dst_path, 'w', encoding='utf-8') as f:
        f.write(out)
    print(f'Wrote commented file to: {dst_path}')

if __name__ == '__main__':
    main()
