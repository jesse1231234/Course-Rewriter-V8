import os
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI

# ============================================================
# Course Rewriter (Canvas + DesignTools/DesignPLUS + OpenAI)
# ============================================================
#
# What this app does:
# - Pull HTML from a target Canvas course (Pages, Assignments, Discussions)
# - Pull example HTML from a model Canvas course
# - Create a compact "style guide" from the model course (LLM pass 1)
# - Rewrite selected items from the target course (LLM pass 2) with strong guardrails
# - Provide side-by-side previews (no diffs) and approvals
# - Push approved HTML back to Canvas
#
# Required secrets/env:
#   CANVAS_BASE_URL
#   CANVAS_API_TOKEN
#   OPENAI_BASE_URL
#   OPENAI_API_KEY
#   OPENAI_MODEL
# ============================================================


# -----------------------------
# DESIGNTOOLS CONTRACT (SYSTEM)
# -----------------------------

DESIGNTOOLS_SYSTEM = """You are a deterministic HTML transformer for Canvas DesignTools (DesignPLUS).
Your job: apply the model-course DesignTools structure faithfully while preserving the original content.

NON-NEGOTIABLE OUTPUT RULES
- Output ONLY HTML (no markdown fences, no commentary).
- Preserve all instructional text verbatim unless the user explicitly requests rewriting text.
- Preserve ALL URLs and Canvas attributes exactly (href/src/id/class/data-*/title/target/rel/style).
- Never delete content; only restructure/wrap to match the model course pattern.
- Do not "clean up" or normalize HTML. Preserve quirky but functioning markup as-is.
- Follow the model course style when it conflicts with generic DesignTools best practice.

DESIGNTOOLS STRUCTURE CONTRACT
A) Wrapper/Header
- Exactly one outer wrapper: <div id="dp-wrapper" class="dp-wrapper ...">.
- Preserve wrapper variant + data-* attributes when used by the model (data-header-class, data-nav-class, data-img-url, data-img-class).
- First child is <header class="dp-header ..."> with <h2 class="dp-heading ...">.
- h2 contains dp-header-pre (often dp-header-pre-1 and dp-header-pre-2) and dp-header-title.
- You may change only the visible text (Module/Week number, title) when required; keep structure.

B) Sectioning
- Use <div class="dp-content-block ..."> for meaningful sections.
- Preserve data-title and data-category attributes if present or implied by the model course.

C) Icons + Headings
- When the model uses icon headings, keep the dp-has-icon pattern and the hidden dp-icon-content span.
- Keep heading levels consistent with the model (h3/h4/h5). Do not skip levels.

D) Embeds and Panels
- Preserve every iframe src EXACTLY.
- If the model course uses dp-embed-wrapper around iframes, then all iframes must be inside dp-embed-wrapper.
- If the model uses panels (dp-panels-wrapper), preserve the same mode (tabs/accordion/expander) and the exact class list.

E) Callouts / Cards
- Preserve dp-callout/card structures and their class lists exactly where present in the model.

F) Discussions
- If the model style uses dp-banner-image on discussions, preserve it (image src + all data-*).
- If the model uses dp-progress-placeholder dp-module-progress-icons, preserve it (typically near the end).

TASK
Apply the model-course DesignTools structure to the provided original HTML and user instructions, with maximal faithfulness.
"""


# -----------------------------
# OpenAI / Azure client
# -----------------------------

def get_ai_client() -> OpenAI:
    """
    OpenAI client configured for Azure AI Foundry "OpenAI-compatible" endpoint.

    Required:
      OPENAI_BASE_URL  e.g. "https://<something>.services.ai.azure.com/openai/v1"
      OPENAI_API_KEY
      OPENAI_MODEL     deployment/model name in Foundry
    """
    base_url = st.secrets.get("OPENAI_BASE_URL", None) or os.getenv("OPENAI_BASE_URL")
    api_key = st.secrets.get("OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY")
    if not base_url or not api_key:
        st.error("Missing OPENAI_BASE_URL / OPENAI_API_KEY (secrets or env).")
        st.stop()
    return OpenAI(base_url=base_url, api_key=api_key)


def get_openai_model_name() -> str:
    model_name = st.secrets.get("OPENAI_MODEL", None) or os.getenv("OPENAI_MODEL")
    if not model_name:
        st.error("Missing OPENAI_MODEL (secrets or env). Set it to your Azure deployment/model name.")
        st.stop()
    return model_name


# -----------------------------
# Canvas helpers
# -----------------------------

def get_canvas_config() -> tuple[str, str]:
    base_url = st.secrets.get("CANVAS_BASE_URL", None) or os.getenv("CANVAS_BASE_URL")
    token = st.secrets.get("CANVAS_API_TOKEN", None) or os.getenv("CANVAS_API_TOKEN")
    if not base_url or not token:
        st.error("Missing CANVAS_BASE_URL / CANVAS_API_TOKEN (secrets or env).")
        st.stop()
    return base_url.rstrip("/"), token


def canvas_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def paginate(url: str, token: str, params: Optional[Dict[str, Any]] = None) -> List[Any]:
    items: List[Any] = []
    headers = canvas_headers(token)
    while url:
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            items.extend(data)
        else:
            items.append(data)

        link = resp.headers.get("Link", "")
        next_url = None
        for part in link.split(","):
            if 'rel="next"' in part:
                next_url = part[part.find("<") + 1 : part.find(">")]
                break
        url = next_url
        params = None
    return items


def get_course(base_url: str, token: str, course_id: str) -> Dict[str, Any]:
    url = f"{base_url}/api/v1/courses/{course_id}"
    resp = requests.get(url, headers=canvas_headers(token))
    resp.raise_for_status()
    return resp.json()


# ---- Pages

def list_pages(base_url: str, token: str, course_id: str) -> List[Dict[str, Any]]:
    url = f"{base_url}/api/v1/courses/{course_id}/pages"
    return paginate(url, token, params={"per_page": 100})


def get_page(base_url: str, token: str, course_id: str, page_url: str) -> Dict[str, Any]:
    url = f"{base_url}/api/v1/courses/{course_id}/pages/{page_url}"
    resp = requests.get(url, headers=canvas_headers(token), params={"include[]": ["body"]})
    resp.raise_for_status()
    return resp.json()


def update_page_html(base_url: str, token: str, course_id: str, page_url: str, html: str) -> None:
    url = f"{base_url}/api/v1/courses/{course_id}/pages/{page_url}"
    resp = requests.put(url, headers=canvas_headers(token), json={"wiki_page": {"body": html}})
    resp.raise_for_status()


# ---- Assignments

def list_assignments(base_url: str, token: str, course_id: str) -> List[Dict[str, Any]]:
    url = f"{base_url}/api/v1/courses/{course_id}/assignments"
    return paginate(url, token, params={"per_page": 100})


def get_assignment(base_url: str, token: str, course_id: str, assignment_id: int) -> Dict[str, Any]:
    url = f"{base_url}/api/v1/courses/{course_id}/assignments/{assignment_id}"
    resp = requests.get(url, headers=canvas_headers(token))
    resp.raise_for_status()
    return resp.json()


def update_assignment_html(base_url: str, token: str, course_id: str, assignment_id: int, html: str) -> None:
    url = f"{base_url}/api/v1/courses/{course_id}/assignments/{assignment_id}"
    resp = requests.put(url, headers=canvas_headers(token), json={"assignment": {"description": html}})
    resp.raise_for_status()


# ---- Discussions

def list_discussions(base_url: str, token: str, course_id: str) -> List[Dict[str, Any]]:
    url = f"{base_url}/api/v1/courses/{course_id}/discussion_topics"
    return paginate(url, token, params={"per_page": 100})


def get_discussion(base_url: str, token: str, course_id: str, topic_id: int) -> Dict[str, Any]:
    url = f"{base_url}/api/v1/courses/{course_id}/discussion_topics/{topic_id}"
    resp = requests.get(url, headers=canvas_headers(token))
    resp.raise_for_status()
    return resp.json()


def update_discussion_html(base_url: str, token: str, course_id: str, topic_id: int, html: str) -> None:
    url = f"{base_url}/api/v1/courses/{course_id}/discussion_topics/{topic_id}"
    resp = requests.put(url, headers=canvas_headers(token), json={"message": html})
    resp.raise_for_status()


# -----------------------------
# Data model
# -----------------------------

@dataclass
class CourseItem:
    kind: str  # "page" | "assignment" | "discussion"
    title: str
    canvas_id: Any  # page_url str, assignment_id int, discussion_id int
    html: str
    url: Optional[str] = None

    # rewrite state
    rewritten_html: Optional[str] = None
    approved: bool = False
    status: str = "original"  # original|rewritten|approved|error
    error: Optional[str] = None


# -----------------------------
# Prompting + Validation
# -----------------------------

_CODE_FENCE_RE = re.compile(r"```(?:html)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_IFRAME_SRC_RE = re.compile(r"<iframe\b[^>]*\bsrc=['\"]([^'\"]+)['\"][^>]*>", re.IGNORECASE)
_CANVAS_API_ENDPOINT_RE = re.compile(r'data-api-endpoint=["\']([^"\']+)["\']', re.IGNORECASE)


def normalize_llm_html(raw: str) -> str:
    if not raw:
        return ""
    m = _CODE_FENCE_RE.search(raw)
    if m:
        raw = m.group(1)
    raw = raw.strip()
    first_lt = raw.find("<")
    last_gt = raw.rfind(">")
    if first_lt != -1 and last_gt != -1 and last_gt > first_lt:
        raw = raw[first_lt:last_gt + 1].strip()
    return raw


def extract_iframe_srcs(html: str) -> List[str]:
    return _IFRAME_SRC_RE.findall(html or "")


def extract_api_endpoints(html: str) -> List[str]:
    return _CANVAS_API_ENDPOINT_RE.findall(html or "")


def detect_model_style_flags(model_context: str) -> Dict[str, Any]:
    ctx = model_context or ""
    return {
        "require_embed_wrapper": ("dp-embed-wrapper" in ctx),
    }


def validate_rewrite(original_html: str, rewritten_html: str, style_flags: Dict[str, Any]) -> List[str]:
    violations: List[str] = []
    o = original_html or ""
    r = rewritten_html or ""

    if r.count('id="dp-wrapper"') != 1:
        violations.append('Output must contain exactly one id="dp-wrapper" wrapper.')

    if "<header" not in r or "dp-header" not in r or "dp-heading" not in r:
        violations.append("Missing required DesignTools header structure (dp-header/dp-heading).")

    # iframe src preservation
    o_iframes = extract_iframe_srcs(o)
    for src in o_iframes:
        if src not in r:
            violations.append(f"Missing original iframe src: {src}")

    # enforce dp-embed-wrapper if model indicates
    if style_flags.get("require_embed_wrapper") and o_iframes:
        if r.lower().count("dp-embed-wrapper") < len(o_iframes):
            violations.append("Model style requires dp-embed-wrapper around iframes; output lacks enough dp-embed-wrapper wrappers.")

    # Preserve Canvas endpoints
    o_eps = set(extract_api_endpoints(o))
    for ep in o_eps:
        if ep and ep not in r:
            violations.append(f"Missing original data-api-endpoint: {ep}")

    # Preserve banner image/progress placeholder if original had them
    if "dp-banner-image" in o and "dp-banner-image" not in r:
        violations.append("Missing dp-banner-image block that existed in the original.")
    if "dp-module-progress-icons" in o and "dp-module-progress-icons" not in r:
        violations.append("Missing dp-module-progress-icons progress placeholder that existed in the original.")

    return violations


def build_style_guide_prompt(model_context: str) -> str:
    return f"""You are analyzing a Canvas course that uses DesignTools / DesignPLUS.
Extract a compact, enforceable "Model Course Style Guide" for rewriting other pages to match it.

Output format:
- A short title line: "Model Course Style Guide"
- Then 12-25 bullet rules (short, strict, implementable)
- Include:
  - Which dp-wrapper variants are used (dp-flat-sections variation-2, dp-rounded-headings, dp-circle-left, etc.)
  - Header structure conventions (dp-heading, dp-header-pre, etc.)
  - How sections are represented (dp-content-block + data-title/data-category usage)
  - Panels usage (tabs/accordion/expander; heading levels h3/h4/h5)
  - iframe handling (dp-embed-wrapper required or not)
  - Discussion-specific patterns (dp-banner-image, progress placeholder)
  - Any consistent color/font classes seen
- Do NOT include explanations. No prose paragraphs.

MODEL COURSE EXAMPLES (may be truncated):
{model_context}
"""


def build_rewrite_prompt(
    item: CourseItem,
    user_instructions: str,
    item_specific_instructions: str,
    model_style_guide: str,
    model_signature_snippets: str,
) -> str:
    user_instructions = (user_instructions or "").strip()
    if not user_instructions:
        user_instructions = "No extra user instructions. Match the model course DesignTools structure and preserve content."

    item_specific_instructions = (item_specific_instructions or "").strip()
    if not item_specific_instructions:
        item_specific_instructions = "No item-specific instructions."

    kind = item.kind
    title = item.title
    html = item.html

    return f"""GLOBAL USER INSTRUCTIONS
{user_instructions}

ITEM-SPECIFIC INSTRUCTIONS (applies ONLY to this item)
{item_specific_instructions}

MODEL COURSE STYLE GUIDE
{model_style_guide}

MODEL SIGNATURE SNIPPETS (use for structural alignment; do not copy content text)
{model_signature_snippets}

TARGET ITEM
- Type: {kind}
- Title: {title}

ORIGINAL HTML
{html}

OUTPUT
Rewrite the ORIGINAL HTML to match the model course structure and styling.
Return ONLY the rewritten HTML.
"""


def build_repair_prompt(original_prompt: str, bad_html: str, violations: List[str]) -> str:
    v = "\n".join([f"- {x}" for x in violations])
    return f"""{original_prompt}

The previous output violated these requirements:
{v}

Fix ONLY these violations. Do NOT rewrite content or make stylistic changes beyond what is required to satisfy the violations.
Return ONLY corrected HTML (no markdown, no commentary).

PREVIOUS OUTPUT (to correct):
{bad_html}
"""


def llm_call(client: OpenAI, user_prompt: str) -> str:
    model_name = get_openai_model_name()
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": DESIGNTOOLS_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    return normalize_llm_html(resp.choices[0].message.content or "")


def rewrite_with_validation(
    client: OpenAI,
    item: CourseItem,
    user_instructions: str,
    item_specific_instructions: str,
    model_style_guide: str,
    model_signature_snippets: str,
    model_context_for_flags: str,
) -> Tuple[str, List[str]]:
    prompt = build_rewrite_prompt(item, user_instructions, item_specific_instructions, model_style_guide, model_signature_snippets)
    style_flags = detect_model_style_flags(model_context_for_flags)

    html1 = llm_call(client, prompt)
    violations = validate_rewrite(item.html, html1, style_flags)
    if not violations:
        return html1, []

    repair_prompt = build_repair_prompt(prompt, html1, violations)
    html2 = llm_call(client, repair_prompt)
    violations2 = validate_rewrite(item.html, html2, style_flags)
    return html2, violations2


# -----------------------------
# UI helpers
# -----------------------------

def html_side_by_side(left_html: str, right_html: str, height: int = 650) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Original")
        components.html(left_html or "<p>(empty)</p>", height=height, scrolling=True)
    with col2:
        st.caption("Proposed rewrite")
        components.html(right_html or "<p>(empty)</p>", height=height, scrolling=True)


def ensure_state():
    st.session_state.setdefault("target_course_id", "")
    st.session_state.setdefault("model_course_id", "")
    st.session_state.setdefault("user_instructions", "")
    st.session_state.setdefault("items", [])  # List[CourseItem]
    st.session_state.setdefault("model_context", "")
    st.session_state.setdefault("model_style_guide", "")
    st.session_state.setdefault("model_signature_snippets", "")
    st.session_state.setdefault("selected_item_key", None)

    # Item-specific prompting
    st.session_state.setdefault("enable_item_prompts", False)
    st.session_state.setdefault("item_prompts", {})  # Dict[item_key, str]
    st.session_state.setdefault(
        "item_prompt_template",
        "Item-specific instructions for this Canvas item (optional).\n"
        "- Preserve all existing wording unless you explicitly change it here.\n"
        "- Apply structural/styling changes to match the model course.\n"
        "- Special requests for THIS item only:\n",
    )


def item_key(item: CourseItem) -> str:
    return f"{item.kind}:{item.canvas_id}"


def default_item_prompt(item: CourseItem) -> str:
    template = st.session_state.get("item_prompt_template", "") or ""
    header = f"{item.kind.upper()} — {item.title}".strip()
    # Keep template stable, but include a short identifier line so prompts are easier to scan.
    return f"{header}\n\n{template}"


def filter_items(items: List[CourseItem], kind_filter: List[str], status_filter: List[str], q: str) -> List[CourseItem]:
    q = (q or "").strip().lower()
    out: List[CourseItem] = []
    for it in items:
        if kind_filter and it.kind not in kind_filter:
            continue
        if status_filter and it.status not in status_filter:
            continue
        if q and (q not in it.title.lower()):
            continue
        out.append(it)
    return out


def build_model_context_from_items(items: List[CourseItem], max_chars: int = 14000) -> str:
    chunks: List[str] = []
    for it in items:
        chunks.append(f"\n\n### {it.kind.upper()}: {it.title}\n{it.html}")
    ctx = "".join(chunks).strip()
    if len(ctx) > max_chars:
        ctx = ctx[:max_chars] + "\n\n[Truncated]"
    return ctx


def build_model_signature_snippets(model_items: List[CourseItem], max_chars: int = 3500) -> str:
    """
    Provide small "signature snippets" to help the model anchor structure.
    We avoid dumping full model HTML here; just a few key openings/patterns.
    """
    sig_parts: List[str] = []
    for it in model_items[:4]:
        html = it.html or ""
        # Take first ~900 chars as signature (header + first blocks)
        sig = html[:900].strip()
        if sig:
            sig_parts.append(f"\n\n--- {it.kind.upper()} EXAMPLE: {it.title} ---\n{sig}")
    sigs = "".join(sig_parts).strip()
    if len(sigs) > max_chars:
        sigs = sigs[:max_chars] + "\n\n[Truncated]"
    return sigs


# -----------------------------
# Load course content
# -----------------------------

def load_course_items(base_url: str, token: str, course_id: str, include_pages: bool, include_assignments: bool, include_discussions: bool) -> List[CourseItem]:
    items: List[CourseItem] = []

    if include_pages:
        pages = list_pages(base_url, token, course_id)
        for p in pages:
            page_url = p.get("url")
            title = p.get("title") or page_url
            if not page_url:
                continue
            body = get_page(base_url, token, course_id, page_url).get("body") or ""
            items.append(CourseItem(kind="page", title=title, canvas_id=page_url, html=body))

    if include_assignments:
        assignments = list_assignments(base_url, token, course_id)
        for a in assignments:
            aid = a.get("id")
            title = a.get("name") or f"Assignment {aid}"
            if not aid:
                continue
            desc = get_assignment(base_url, token, course_id, int(aid)).get("description") or ""
            items.append(CourseItem(kind="assignment", title=title, canvas_id=int(aid), html=desc))

    if include_discussions:
        discussions = list_discussions(base_url, token, course_id)
        for d in discussions:
            did = d.get("id")
            title = d.get("title") or f"Discussion {did}"
            if not did:
                continue
            msg = get_discussion(base_url, token, course_id, int(did)).get("message") or ""
            items.append(CourseItem(kind="discussion", title=title, canvas_id=int(did), html=msg))

    return items


# -----------------------------
# App
# -----------------------------

def main():
    st.set_page_config(page_title="Course Rewriter (DesignTools)", layout="wide")
    ensure_state()

    base_url, token = get_canvas_config()
    client = get_ai_client()

    st.title("Course Rewriter (Canvas + DesignTools + LLM)")

    tabs = st.tabs(["Connect + Load", "Model Course", "Rewrite", "Review", "Publish"])

    # -----------------
    # Tab 1: Connect + Load
    # -----------------
    with tabs[0]:
        st.subheader("Connect + Load")

        colA, colB = st.columns(2)
        with colA:
            st.text_input("Target course ID (to rewrite)", key="target_course_id")
        with colB:
            st.text_input("Model course ID (style reference)", key="model_course_id")

        st.text_area(
            "Global user instructions (applied to all rewrites)",
            key="user_instructions",
            height=140,
            placeholder="Example: Keep all wording exactly; restructure into DesignTools blocks like the model; preserve all file links and iframe src.",
        )

        st.markdown("**Content types**")
        c1, c2, c3 = st.columns(3)
        with c1:
            include_pages = st.checkbox("Pages", value=True)
        with c2:
            include_assignments = st.checkbox("Assignments", value=True)
        with c3:
            include_discussions = st.checkbox("Discussions", value=True)

        load_btn = st.button("Load target course items", type="primary")
        if load_btn:
            cid = (st.session_state["target_course_id"] or "").strip()
            if not cid:
                st.error("Enter a target course ID.")
            else:
                with st.spinner("Loading target course content from Canvas..."):
                    try:
                        course = get_course(base_url, token, cid)
                        st.session_state["target_course_name"] = course.get("name") or cid
                        items = load_course_items(base_url, token, cid, include_pages, include_assignments, include_discussions)
                        st.session_state["items"] = items
                        # Initialize item-specific prompts for newly loaded items
                        item_prompts: Dict[str, str] = st.session_state.get("item_prompts", {}) or {}
                        for it in items:
                            k = item_key(it)
                            item_prompts.setdefault(k, default_item_prompt(it))
                        st.session_state["item_prompts"] = item_prompts
                        st.success(f"Loaded {len(items)} items from target course.")
                    except Exception as e:
                        st.error(f"Failed to load target course: {e}")

        if st.session_state.get("items"):
            st.info(f"Current target items loaded: {len(st.session_state['items'])}")

    # -----------------
    # Tab 2: Model Course
    # -----------------
    with tabs[1]:
        st.subheader("Model Course")

        model_id = (st.session_state["model_course_id"] or "").strip()
        if not model_id:
            st.warning("Enter a Model course ID in the first tab.")
        else:
            st.markdown("Load a sample of model course items to infer patterns.")
            colA, colB, colC = st.columns(3)
            with colA:
                sample_pages = st.number_input("Sample pages", min_value=0, max_value=50, value=6, step=1)
            with colB:
                sample_assignments = st.number_input("Sample assignments", min_value=0, max_value=50, value=3, step=1)
            with colC:
                sample_discussions = st.number_input("Sample discussions", min_value=0, max_value=50, value=3, step=1)

            if st.button("Load model samples + build style guide", type="primary"):
                with st.spinner("Loading model course samples..."):
                    try:
                        model_items_all = load_course_items(
                            base_url,
                            token,
                            model_id,
                            include_pages=(sample_pages > 0),
                            include_assignments=(sample_assignments > 0),
                            include_discussions=(sample_discussions > 0),
                        )
                        # Keep only up to requested counts in each kind
                        model_items: List[CourseItem] = []
                        for kind, n in [("page", int(sample_pages)), ("assignment", int(sample_assignments)), ("discussion", int(sample_discussions))]:
                            if n <= 0:
                                continue
                            subset = [x for x in model_items_all if x.kind == kind][:n]
                            model_items.extend(subset)

                        model_context = build_model_context_from_items(model_items, max_chars=14000)
                        st.session_state["model_context"] = model_context
                        st.session_state["model_signature_snippets"] = build_model_signature_snippets(model_items, max_chars=3500)

                        # LLM pass: style guide
                        with st.spinner("Generating model style guide (LLM)..."):
                            style_prompt = build_style_guide_prompt(model_context)
                            style_guide = llm_call(client, style_prompt)
                            st.session_state["model_style_guide"] = style_guide

                        st.success("Model style guide ready.")
                    except Exception as e:
                        st.error(f"Failed to build model guide: {e}")

            if st.session_state.get("model_style_guide"):
                st.markdown("### Model Course Style Guide")
                st.code(st.session_state["model_style_guide"], language="text")

            if st.session_state.get("model_signature_snippets"):
                with st.expander("Model signature snippets (what the LLM sees as structure anchors)"):
                    st.code(st.session_state["model_signature_snippets"], language="html")

    # -----------------
    # Tab 3: Rewrite (queue + batch)
    # -----------------
    with tabs[2]:
        st.subheader("Rewrite")

        items: List[CourseItem] = st.session_state.get("items", [])
        if not items:
            st.warning("Load target items in the first tab.")
        elif not st.session_state.get("model_style_guide"):
            st.warning("Build the model style guide in the Model Course tab.")
        else:
            kinds = sorted({it.kind for it in items})
            status_values = ["original", "rewritten", "approved", "error"]

            col1, col2, col3 = st.columns([2, 2, 3])
            with col1:
                kind_filter = st.multiselect("Type filter", options=kinds, default=kinds)
            with col2:
                status_filter = st.multiselect("Status filter", options=status_values, default=["original", "error"])
            with col3:
                q = st.text_input("Search title", value="")

            filtered = filter_items(items, kind_filter, status_filter, q)
            st.caption(f"Showing {len(filtered)} items")

            # Item-specific prompting (optional)
            st.checkbox("Enable item-specific prompting", key="enable_item_prompts")
            if st.session_state.get("enable_item_prompts", False):
                st.caption("Item-specific prompts are appended to the global instructions when rewriting each item.")
                with st.expander("Item-specific prompts", expanded=False):
                    scope = st.radio(
                        "Edit prompts for",
                        options=["Selected items", "All visible items"],
                        horizontal=True,
                        help="Use Selected items to keep the list manageable.",
                    )
                    item_prompts: Dict[str, str] = st.session_state.get("item_prompts", {}) or {}

                    if scope == "Selected items":
                        keys_to_edit = selected_keys or []
                    else:
                        keys_to_edit = [item_key(it) for it in filtered]

                    if not keys_to_edit:
                        st.info("Select items first (or switch scope to All visible items).")
                    else:
                        colX, colY = st.columns(2)
                        with colX:
                            if st.button("Fill missing prompts with template", key="fill_missing_prompts"):
                                for k in keys_to_edit:
                                    it0 = next((x for x in items if item_key(x) == k), None)
                                    if it0 is None:
                                        continue
                                    if not (item_prompts.get(k) or "").strip():
                                        item_prompts[k] = default_item_prompt(it0)
                                st.session_state["item_prompts"] = item_prompts
                                st.success("Filled missing prompts.")
                        with colY:
                            if st.button("Clear prompts for this scope", key="clear_scope_prompts"):
                                for k in keys_to_edit:
                                    if k in item_prompts:
                                        item_prompts[k] = ""
                                st.session_state["item_prompts"] = item_prompts
                                st.success("Cleared prompts.")

                        st.divider()

                        # Avoid rendering hundreds of textareas
                        max_edit = 30
                        if len(keys_to_edit) > max_edit:
                            st.warning(f"Showing first {max_edit} prompts (out of {len(keys_to_edit)}). Use filters to narrow the list.")
                        for k in keys_to_edit[:max_edit]:
                            it0 = next((x for x in items if item_key(x) == k), None)
                            if it0 is None:
                                continue
                            label = f"{it0.title}  ({it0.kind})"
                            default_val = item_prompts.get(k)
                            if default_val is None:
                                default_val = default_item_prompt(it0)
                                item_prompts[k] = default_val
                            item_prompts[k] = st.text_area(
                                label,
                                value=default_val,
                                height=120,
                                key=f"item_prompt__{k}",
                            )
                        st.session_state["item_prompts"] = item_prompts

            # selection
            selected_keys = st.multiselect(
                "Select items to rewrite (by title)",
                options=[item_key(it) for it in filtered],
                format_func=lambda k: next((x.title for x in items if item_key(x) == k), k),
            )

            colA, colB = st.columns(2)
            with colA:
                run_selected = st.button("Rewrite selected", type="primary", disabled=(len(selected_keys) == 0))
            with colB:
                run_all_visible = st.button("Rewrite all visible", disabled=(len(filtered) == 0))

            if run_selected or run_all_visible:
                if run_all_visible:
                    targets = filtered
                else:
                    targets = [it for it in items if item_key(it) in set(selected_keys)]

                user_instructions = st.session_state.get("user_instructions", "")
                model_style_guide = st.session_state.get("model_style_guide", "")
                model_sig = st.session_state.get("model_signature_snippets", "")
                model_context_for_flags = st.session_state.get("model_context", "")

                enable_item_prompts = bool(st.session_state.get("enable_item_prompts", False))
                item_prompts: Dict[str, str] = st.session_state.get("item_prompts", {}) or {}

                progress = st.progress(0)
                errors = 0
                for idx, it in enumerate(targets, start=1):
                    try:
                        with st.spinner(f"Rewriting: {it.title}"):
                            if enable_item_prompts:
                                k = item_key(it)
                                item_specific_instructions = (item_prompts.get(k) or "").strip()
                                if not item_specific_instructions:
                                    item_specific_instructions = default_item_prompt(it)
                                    item_prompts[k] = item_specific_instructions
                            else:
                                item_specific_instructions = ""

                            new_html, violations = rewrite_with_validation(
                                client=client,
                                item=it,
                                user_instructions=user_instructions,
                                item_specific_instructions=item_specific_instructions,
                                model_style_guide=model_style_guide,
                                model_signature_snippets=model_sig,
                                model_context_for_flags=model_context_for_flags,
                            )
                            it.rewritten_html = new_html
                            if violations:
                                it.status = "error"
                                it.error = "Validation issues:\n" + "\n".join(violations)
                                errors += 1
                            else:
                                it.status = "rewritten"
                                it.error = None
                    except Exception as e:
                        it.status = "error"
                        it.error = str(e)
                        errors += 1
                    progress.progress(idx / max(1, len(targets)))

                st.session_state["item_prompts"] = item_prompts

                st.session_state["items"] = items
                if errors:
                    st.warning(f"Rewrite finished with {errors} items in error state. Review them in the Review tab.")
                else:
                    st.success("Rewrite finished. Review results in the Review tab.")

    # -----------------
    # Tab 4: Review (detail panel)
    # -----------------
    with tabs[3]:
        st.subheader("Review")

        items: List[CourseItem] = st.session_state.get("items", [])
        if not items:
            st.warning("Load target items first.")
        else:
            # Quick stats
            counts = {s: 0 for s in ["original", "rewritten", "approved", "error"]}
            for it in items:
                counts[it.status] = counts.get(it.status, 0) + 1
            st.write(f"Original: {counts['original']}  |  Rewritten: {counts['rewritten']}  |  Approved: {counts['approved']}  |  Errors: {counts['error']}")

            col1, col2, col3 = st.columns([2, 2, 3])
            with col1:
                kind_filter = st.multiselect("Type", options=sorted({it.kind for it in items}), default=sorted({it.kind for it in items}))
            with col2:
                status_filter = st.multiselect("Status", options=["original", "rewritten", "approved", "error"], default=["rewritten", "error", "approved"])
            with col3:
                q = st.text_input("Search", value="", key="review_search")

            filtered = filter_items(items, kind_filter, status_filter, q)
            st.caption(f"{len(filtered)} items match filters")

            # pick one item
            selected = st.selectbox(
                "Pick an item to review",
                options=[item_key(it) for it in filtered],
                format_func=lambda k: next((x.title for x in items if item_key(x) == k), k),
                index=0 if filtered else None,
            ) if filtered else None

            if selected:
                it = next(x for x in items if item_key(x) == selected)

                st.markdown(f"### {it.title}")
                st.caption(f"Type: {it.kind} | Status: {it.status}")

                # Item-specific prompt (can be used even if item-specific prompting is off globally)
                k_item = item_key(it)
                item_prompts: Dict[str, str] = st.session_state.get("item_prompts", {}) or {}
                with st.expander("Item-specific prompt", expanded=False):
                    current_prompt = item_prompts.get(k_item)
                    if current_prompt is None:
                        current_prompt = default_item_prompt(it)
                        item_prompts[k_item] = current_prompt

                    new_prompt = st.text_area(
                        "Prompt for this item",
                        value=current_prompt,
                        height=140,
                        key=f"review_item_prompt__{k_item}",
                        help="This prompt is appended to the global instructions when you re-run a rewrite for this item.",
                    )
                    item_prompts[k_item] = new_prompt
                    st.session_state["item_prompts"] = item_prompts

                    can_rerun = bool(st.session_state.get("model_style_guide"))
                    if st.button("Re-run rewrite for this item", type="primary", disabled=not can_rerun):
                        with st.spinner("Rewriting this item..."):
                            try:
                                user_instructions = st.session_state.get("user_instructions", "")
                                model_style_guide = st.session_state.get("model_style_guide", "")
                                model_sig = st.session_state.get("model_signature_snippets", "")
                                model_context_for_flags = st.session_state.get("model_context", "")

                                new_html, violations = rewrite_with_validation(
                                    client=client,
                                    item=it,
                                    user_instructions=user_instructions,
                                    item_specific_instructions=new_prompt,
                                    model_style_guide=model_style_guide,
                                    model_signature_snippets=model_sig,
                                    model_context_for_flags=model_context_for_flags,
                                )
                                it.rewritten_html = new_html
                                it.approved = False
                                if violations:
                                    it.status = "error"
                                    it.error = "Validation issues:\n" + "\n".join(violations)
                                else:
                                    it.status = "rewritten"
                                    it.error = None
                                st.session_state["items"] = items
                                st.success("Rewrite updated for this item.")
                            except Exception as e:
                                it.status = "error"
                                it.error = str(e)
                                st.session_state["items"] = items
                                st.error(f"Failed to rewrite item: {e}")

                if it.status == "error" and it.error:
                    st.error(it.error)

                # approve toggle
                approve = st.checkbox("Approve this rewrite", value=it.approved or (it.status == "approved"))
                it.approved = bool(approve) and bool(it.rewritten_html)
                if it.approved:
                    it.status = "approved"
                elif it.status == "approved":
                    it.status = "rewritten" if it.rewritten_html else "original"

                # show previews
                if not it.rewritten_html:
                    st.info("No rewrite generated yet.")
                else:
                    html_side_by_side(it.html, it.rewritten_html, height=650)

                st.session_state["items"] = items

            # batch approvals
            st.markdown("#### Batch actions")
            colA, colB = st.columns(2)
            with colA:
                if st.button("Approve all rewritten in current filter"):
                    for it in filtered:
                        if it.rewritten_html and it.status == "rewritten":
                            it.approved = True
                            it.status = "approved"
                    st.session_state["items"] = items
                    st.success("Approved all rewritten items in current filter.")
            with colB:
                if st.button("Unapprove all in current filter"):
                    for it in filtered:
                        it.approved = False
                        if it.status == "approved":
                            it.status = "rewritten" if it.rewritten_html else "original"
                    st.session_state["items"] = items
                    st.success("Unapproved items in current filter.")

    # -----------------
    # Tab 5: Publish (push to Canvas)
    # -----------------
    with tabs[4]:
        st.subheader("Publish")

        items: List[CourseItem] = st.session_state.get("items", [])
        cid = (st.session_state.get("target_course_id") or "").strip()
        if not items or not cid:
            st.warning("Load target items first.")
        else:
            approved = [it for it in items if it.approved and it.rewritten_html]
            st.write(f"Approved items ready to publish: **{len(approved)}**")

            if len(approved) == 0:
                st.info("Approve at least one item in the Review tab before publishing.")
            else:
                dry_run = st.checkbox("Dry run (do not write to Canvas)", value=True)
                if st.button("Publish approved to Canvas", type="primary"):
                    errors: List[Tuple[str, str]] = []
                    with st.spinner("Publishing to Canvas..."):
                        for it in approved:
                            try:
                                if dry_run:
                                    continue
                                if it.kind == "page":
                                    update_page_html(base_url, token, cid, str(it.canvas_id), it.rewritten_html or "")
                                elif it.kind == "assignment":
                                    update_assignment_html(base_url, token, cid, int(it.canvas_id), it.rewritten_html or "")
                                elif it.kind == "discussion":
                                    update_discussion_html(base_url, token, cid, int(it.canvas_id), it.rewritten_html or "")
                                else:
                                    raise ValueError(f"Unknown item kind: {it.kind}")
                            except Exception as e:
                                errors.append((it.title, str(e)))

                    if dry_run:
                        st.success("Dry run complete. No changes were written to Canvas.")
                    elif errors:
                        st.error("Some items failed to publish:")
                        for title, msg in errors:
                            st.write(f"- **{title}**: {msg}")
                    else:
                        st.success("All approved items successfully written back to Canvas.")


if __name__ == "__main__":
    main()
