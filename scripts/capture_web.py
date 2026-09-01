"""Drive the web UI through one real question and record it as a GIF.

A still of the idle page shows nothing, and a still of the finished page shows only the
end — no prompt, no plan, no activity filling up. So this types a question, captures the
page each time it visibly changes, and stitches the frames into an animation.

Only `<main>` is captured, never the whole window: the sidebar lists the operator's real
session history, which has no business in a published screenshot.

Doubles as the end-to-end check `CLAUDE.md` asks for — unit tests never load `app.js`, and
a silent breakage of this page is exactly what the suite missed before.

Playwright lives in its own uv tool environment, so run this with that interpreter, with a
server already up (`helioai serve --web`):

    ~/.local/share/uv/tools/playwright/bin/python scripts/capture_web.py

Environment:
    HELIOAI_WEB_URL    default http://127.0.0.1:7890
    CAPTURE_OUT        final still,  default docs/assets/web-ui.png
    CAPTURE_GIF        animation,    default docs/assets/web-demo.gif
    CAPTURE_QUERY      the question to ask
    CAPTURE_TIMEOUT_S  how long to wait for the answer (default 300)
    CAPTURE_WIDTH      GIF width in px (default 1000); frames are downscaled to it
"""

from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

URL = os.environ.get("HELIOAI_WEB_URL", "http://127.0.0.1:7890")
OUT = Path(os.environ.get("CAPTURE_OUT", "docs/assets/web-ui.png"))
GIF_OUT = Path(os.environ.get("CAPTURE_GIF", "docs/assets/web-demo.gif"))
QUERY = os.environ.get(
    "CAPTURE_QUERY",
    "Plot the Wind magnetic field magnitude at 3 s cadence for 2015-03-17 03:30 to 05:00, "
    "and mark the shock arrival",
)
TIMEOUT_S = int(os.environ.get("CAPTURE_TIMEOUT_S", "300"))
GIF_WIDTH = int(os.environ.get("CAPTURE_WIDTH", "1000"))
MAX_FRAMES = 28
FRAME_MS = 1400
LAST_FRAME_MS = 4000


def _write_gif(frames: list[bytes], out: Path, width: int = GIF_WIDTH) -> None:
    """Stitch PNG bytes into a looping GIF, downscaled and palette-quantised.

    Downscaling first matters more than it looks: a 1440-wide true-colour frame
    quantises to a visibly dithered mess, and thirty of them make a file GitHub
    refuses to animate.
    """
    from PIL import Image

    raws = [Image.open(io.BytesIO(b)).convert("RGB") for b in frames]

    # Frames are not all the same shape — the closing one spans the code panel too — and a
    # GIF needs one canvas for every frame. Scale each to the common width, then pad to the
    # tallest, rather than stretching anything to fit.
    scaled = [
        im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
        if im.width != width
        else im
        for im in raws
    ]
    canvas = (width, max(im.height for im in scaled))
    images = []
    for im in scaled:
        if im.size != canvas:
            padded = Image.new("RGB", canvas, (13, 17, 23))
            padded.paste(im, (0, 0))
            im = padded
        images.append(im.quantize(colors=128, method=Image.MEDIANCUT))

    durations = [FRAME_MS] * len(images)
    durations[-1] = LAST_FRAME_MS
    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        out,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "playwright is not installed in this interpreter — see the docstring.", file=sys.stderr
        )
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))

        page.goto(URL, wait_until="networkidle")
        main_el = page.locator("main").first

        # Expand the dock before sending: it starts collapsed, and watching the reasoning
        # chain fill up live is the point of animating this at all.
        try:
            page.click("#ad-header", timeout=2000)
        except Exception:
            pass

        # Not a multi-selector: #dev-token-input comes first in the DOM and would swallow
        # the question.
        page.fill("#input", QUERY)
        frames = [main_el.screenshot()]
        page.click("#btn-send")

        def signature() -> tuple:
            return (
                page.locator("#ad-body > *").count(),
                page.locator("#chat-area > *").count(),
                page.locator("#chat-area img").count(),
            )

        # Completion signal: Cancel is visible for the duration of a run and hidden when it
        # ends. Waiting on "the dock stopped growing" instead reports success during the
        # first LLM turn, when the dock is still legitimately empty — that produced a
        # capture of a half-finished run that read as a bug in the UI.
        page.wait_for_selector("#btn-cancel:visible", timeout=30_000)

        # Sampling on a timer would yield a hundred near-identical frames of an agent
        # thinking. Capture on *change*: the visible beats of a run are exactly the moments
        # a dock row, a chat bubble or a figure appears.
        deadline, last_sig = time.time() + TIMEOUT_S, None
        while time.time() < deadline:
            sig = signature()
            if sig != last_sig and len(frames) < MAX_FRAMES:
                frames.append(main_el.screenshot())
                last_sig = sig
            if page.locator("#btn-cancel").first.is_hidden():
                break
            page.wait_for_timeout(1000)

        page.wait_for_timeout(1500)
        last_count = page.locator("#ad-body > *").count()

        # The dock collapses itself when the run ends, so expanding it once up front is not
        # enough — the closing frame, the one a reader lingers on, would hide the reasoning
        # chain that justifies animating this at all.
        if "collapsed" in (page.locator("#activity-dock").get_attribute("class") or ""):
            page.click("#ad-header")
            page.wait_for_timeout(800)
        frames.append(main_el.screenshot())

        # Finish on the generated code. The panel is a third column, a sibling of <main>
        # rather than a child, so an element screenshot of main would crop it away —
        # capture a clip spanning both instead, still starting past the sidebar.
        chips = page.locator(".artifact-code")
        if chips.count():
            chips.last.click()
            page.wait_for_timeout(1200)
            box = main_el.bounding_box()
            panel = page.locator("#code-panel").bounding_box()
            if box and panel:
                right = max(box["x"] + box["width"], panel["x"] + panel["width"])
                frames.append(
                    page.screenshot(
                        clip={
                            "x": box["x"],
                            "y": box["y"],
                            "width": right - box["x"],
                            "height": box["height"],
                        }
                    )
                )
        OUT.write_bytes(frames[-1])
        browser.close()

    _write_gif(frames, GIF_OUT)
    print(f"still: {OUT} ({OUT.stat().st_size // 1024} KB), {last_count} timeline rows")
    print(f"gif:   {GIF_OUT} ({GIF_OUT.stat().st_size // 1024} KB), {len(frames)} frames")
    if errors:
        # A page that throws still screenshots fine; saying so beats shipping a picture of
        # a half-rendered UI.
        print("JS errors on the page:", file=sys.stderr)
        for e in errors[:5]:
            print(f"  {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
