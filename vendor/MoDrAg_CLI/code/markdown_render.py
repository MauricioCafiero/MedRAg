"""
Unicode preprocessor for LLM Markdown responses before rich rendering.

rich (15.x) / markdown-it-py render CommonMark but have no math renderer and
silently strip inline HTML tags, so a model response containing

    Cys<sub>14</sub> and Zn<sup>2+</sup>; affinity K_d ~ e^{-dG/RT}

would render as ``Cys14 and Zn2+`` (subscript meaning lost) with any ``$...$``
math shown as literal delimiters. This module pre-converts, before handing the
string to ``rich.markdown.Markdown``:

  * <sub>...</sub> / <sup>...</sup> -> Unicode sub/superscripts when every char
    of the content has a Unicode form (digits, signs, and the letters that have
    subscript/superscript glyphs). If any char lacks a form, the original tag
    markup is left untouched rather than silently dropped.
  * $...$ and $$...$$ math -> delimiters stripped; ``_{...}`` / ``^{...}``
    (and single-char ``_x`` / ``^x``) converted to Unicode sub/superscripts
    where the whole group is convertible, otherwise left literal.

Fenced code (```...```) and inline code (`...`) are protected -- their contents
are never touched, so SMILES / code containing ``_``, ``^``, ``$`` or ``<sub>``
is passed through verbatim. ``_`` and ``^`` outside math delimiters are also left
alone (Markdown emphasis / literal text handles those).

Design choice: single-char and fully-mappable brace groups are converted;
multi-character groups with no clean Unicode (e.g. ``e^{\\Delta G/RT}``,
uppercase letters, ``/``) stay literal. True LaTeX rendering isn't possible in a
terminal; this gives readable formulas without garbling the un-mappable cases.
"""

import re

# Unicode subscript/superscript maps. Only characters that have a real Unicode
# sub/superscript codepoint are included; everything else (uppercase letters,
# '/', '\\', Greek, ...) intentionally absent so the group falls back to literal.
_SUB = {
    '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
    '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
    'a': 'ₐ', 'e': 'ₑ', 'h': 'ₕ', 'i': 'ᵢ', 'j': 'ⱼ',
    'k': 'ₖ', 'l': 'ₗ', 'm': 'ₘ', 'n': 'ₙ', 'o': 'ₒ',
    'p': 'ₚ', 'r': 'ᵣ', 's': 'ₛ', 't': 'ₜ', 'u': 'ᵤ',
    'v': 'ᵥ', 'x': 'ₓ',
    '+': '₊', '-': '₋', '=': '₌', '(': '₍', ')': '₎',
}
_SUP = {
    '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
    '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
    'a': 'ᵃ', 'b': 'ᵇ', 'c': 'ᶜ', 'd': 'ᵈ', 'e': 'ᵉ',
    'f': 'ᶠ', 'g': 'ᵍ', 'h': 'ʰ', 'i': 'ⁱ', 'j': 'ʲ',
    'k': 'ᵏ', 'l': 'ˡ', 'm': 'ᵐ', 'n': 'ⁿ', 'o': 'ᵒ',
    'p': 'ᵖ', 'r': 'ʳ', 's': 'ˢ', 't': 'ᵗ', 'u': 'ᵘ',
    'v': 'ᵛ', 'w': 'ʷ', 'x': 'ˣ', 'y': 'ʸ', 'z': 'ᶻ',
    '+': '⁺', '-': '⁻', '=': '⁼', '(': '⁽', ')': '⁾',
}


def _convert_group(content, mapping):
    """Return content with every char mapped, or None if any char is unmappable.

    A group is all-or-nothing: if one character has no Unicode sub/superscript
    form, the whole group is left literal rather than producing a half-converted
    glyph soup.
    """
    out = []
    for ch in content:
        mapped = mapping.get(ch)
        if mapped is None:
            return None
        out.append(mapped)
    return ''.join(out)


def _convert_math(content):
    """Convert ``_x`` / ``^x`` / ``_{...}`` / ``^{...}`` in math content to Unicode.

    Leaves everything else (including LaTeX commands like ``\\Delta``) literal.
    """
    out = []
    i, n = 0, len(content)
    while i < n:
        ch = content[i]
        if ch in ('_', '^') and i + 1 < n:
            mapping = _SUB if ch == '_' else _SUP
            if content[i + 1] == '{':
                j = content.find('}', i + 2)
                if j != -1:
                    inner = content[i + 2:j]
                    conv = _convert_group(inner, mapping)
                    if conv is not None:
                        out.append(conv)
                        i = j + 1
                        continue
                    # un-mappable group -> keep literal, braces included
                    out.append(content[i:j + 1])
                    i = j + 1
                    continue
            else:
                single = content[i + 1]
                conv = _convert_group(single, mapping)
                if conv is not None:
                    out.append(conv)
                    i += 2
                    continue
        out.append(ch)
        i += 1
    return ''.join(out)


# Single-pass tokenizer. Alternation order matters: fenced code and inline code
# are matched first (protected verbatim), then block math ``$$..$$`` before
# inline math ``$..$`` so the latter can't eat a block, then sub/sup tags.
_TOKEN = re.compile(
    r'(?P<fence>```.*?```)'                       # fenced code (DOTALL)
    r'|(?P<icode>`[^`]*`)'                        # inline code
    r'|(?P<mathb>\$\$.*?\$\$)'                     # block math (DOTALL)
    r'|(?P<mathi>\$[^$\n]+?\$)'                   # inline math (single line)
    r'|(?P<sub><sub>(?P<subc>.*?)</sub>)'         # subscript tag
    r'|(?P<sup><sup>(?P<supc>.*?)</sup>)',        # superscript tag
    re.DOTALL,
)


def _replace(match):
    if match.group('fence') is not None or match.group('icode') is not None:
        return match.group(0)                       # protected: code verbatim
    if match.group('mathb') is not None:
        inner = match.group('mathb')[2:-2]           # strip $$
        return _convert_math(inner)
    if match.group('mathi') is not None:
        inner = match.group('mathi')[1:-1]           # strip $
        return _convert_math(inner)
    if match.group('sub') is not None:
        conv = _convert_group(match.group('subc'), _SUB)
        return conv if conv is not None else match.group('sub')
    if match.group('sup') is not None:
        conv = _convert_group(match.group('supc'), _SUP)
        return conv if conv is not None else match.group('sup')
    return match.group(0)


def to_unicode_markdown(text):
    """Pre-process LLM Markdown so rich renders readable formulas/tags.

    Pass the result to ``rich.markdown.Markdown``. Returns the input unchanged
    if it isn't a string.
    """
    if not isinstance(text, str):
        return text
    return _TOKEN.sub(_replace, text)