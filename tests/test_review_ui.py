"""Browser integration tests for the local review workflow."""

import csv
import io
import json
import re
import socket
import sys
import unittest
from importlib.util import find_spec
from threading import Thread
from time import monotonic, sleep

from src.config import BENCHMARK_METRICS_PATH
from src.data_loader import load_dataset


@unittest.skipUnless(
    all(find_spec(name) for name in ("fastapi", "uvicorn", "playwright")),
    "Install the review application and Playwright to run browser tests",
)
class ReviewInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import uvicorn
        from playwright.sync_api import sync_playwright

        from app.review_ui import app

        cls.dataset = load_dataset()
        listener = socket.socket()
        cls.addClassCleanup(listener.close)
        listener.bind(("127.0.0.1", 0))
        cls.url = f"http://127.0.0.1:{listener.getsockname()[1]}"
        server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
        thread = Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()

        def shutdown():
            server.should_exit = True
            thread.join(timeout=15)

        cls.addClassCleanup(shutdown)
        deadline = monotonic() + 120
        while not server.started and thread.is_alive() and monotonic() < deadline:
            sleep(0.05)
        if not server.started:
            raise RuntimeError("Review server did not start")

        playwright = sync_playwright().start()
        cls.addClassCleanup(playwright.stop)
        cls.browser = playwright.chromium.launch(
            channel="msedge" if sys.platform == "win32" else None,
        )
        cls.addClassCleanup(cls.browser.close)

    def setUp(self):
        self.context = self.browser.new_context(accept_downloads=True)
        self.addCleanup(self.context.close)
        self.page = self.context.new_page()
        self.page.set_default_timeout(15_000)
        response = self.page.goto(self.url)
        self.assertEqual(response.status, 200)

    def analyze(self, text):
        self.page.locator("#note-select:not([disabled])").wait_for()
        self.page.get_by_role("button", name="Paste Custom Note").click()
        self.page.get_by_label("Clinical note text").fill(text)
        self.page.get_by_role("button", name="Analyze note").click()

    def test_compact_workspace_and_synced_navigation_at_1366x768(self):
        from playwright.sync_api import expect

        self.page.set_viewport_size({"width": 1366, "height": 768})
        self.page.emulate_media(color_scheme="dark", reduced_motion="reduce")
        theme = self.page.locator("#theme-toggle")
        expect(theme).to_have_text("Theme: Dark")
        theme.press("Space")
        expect(theme).to_have_attribute("aria-pressed", "false")
        self.page.reload()
        expect(theme).to_have_text("Theme: Light")
        expect(self.page.locator("body")).to_have_css("background-color", "rgb(244, 244, 242)")
        self.analyze(("Heart failure. " + "Today " * 60) * 24)
        rows = self.page.locator("#review-rows tr")
        marks = self.page.locator("#note-text mark")
        expect(rows).to_have_count(24)
        expect(self.page.locator("#total-count")).to_have_text("24")
        self.assertLessEqual(self.page.locator(".toolbar").bounding_box()["height"], 64)
        self.assertTrue(self.page.evaluate("""() => {
            const panes = [document.querySelector("#note-text"), document.querySelector(".table-wrap")];
            return document.documentElement.scrollWidth === innerWidth
                && document.documentElement.scrollHeight === innerHeight
                && panes.every(pane => pane.scrollWidth === pane.clientWidth && pane.scrollHeight > pane.clientHeight);
        }"""))
        expect(rows.last.locator("td").nth(3)).to_have_text(re.compile(r"^\d{1,3}%$"))
        marks.last.get_by_role("button").click()
        expect(rows.last).to_have_class("is-current")
        expect(rows.last.get_by_role("button", name="Reject")).to_be_in_viewport()
        expect(self.page.get_by_role("columnheader", name="SNOMED CT ID")).to_be_in_viewport()

        self.page.locator("#note-text").evaluate("(pane) => { pane.scrollTop = 0; }")
        rows.last.locator("td").nth(1).click()
        expect(marks.last).to_have_class("is-current")
        expect(marks.last).to_be_in_viewport()
        rows.first.get_by_role("button", name="Show note mention for Heart failure").press("Enter")
        expect(marks.first).to_have_class("is-current")
        expect(marks.first.get_by_role("button")).to_be_focused()
        theme.click()
        self.page.reload()
        expect(theme).to_have_text("Theme: Dark")
        expect(self.page.locator("body")).to_have_css("background-color", "rgb(23, 24, 27)")

    def test_redesign_breakpoints_and_touch_targets(self):
        from playwright.sync_api import expect

        self.analyze("Heart failure. Renal failure.")
        expect(self.page.locator("#review-rows tr")).to_have_count(2)
        for theme in ("light", "dark"):
            self.page.evaluate("theme => document.documentElement.dataset.theme = theme", theme)
            for width in (375, 768, 1024, 1440):
                with self.subTest(theme=theme, width=width):
                    self.page.set_viewport_size({"width": width, "height": 900})
                    self.assertTrue(self.page.evaluate("""() => {
                        const controls = document.querySelectorAll(
                            '.toolbar button:not(#export-options button), .toolbar select, .candidate-select, .choices button, mark button, .mention-link'
                        );
                        return document.documentElement.scrollWidth === innerWidth &&
                            [...controls].every(element => {
                                const box = element.getBoundingClientRect();
                                return box.width >= 44 && box.height >= 44;
                            });
                    }"""))
                    review_pane = self.page.locator(".table-wrap").bounding_box()
                    first_row = self.page.locator("#review-rows tr").first.bounding_box()
                    self.assertGreater(first_row["width"], review_pane["width"] * 0.95)
                    self.page.get_by_role("button", name="Metrics", exact=True).click()
                    dialog = self.page.get_by_role("dialog", name="Validation Metrics")
                    expect(dialog).to_be_visible()
                    expect(self.page.locator("#metrics-table")).to_be_visible()
                    self.assertTrue(dialog.evaluate("e => e.scrollWidth === e.clientWidth"))
                    self.page.keyboard.press("Escape")
                    expect(dialog).to_be_hidden()

    def test_mention_popup_dismissal_and_review(self):
        from playwright.sync_api import expect

        self.analyze("Heart failure. Renal failure.")
        mentions = self.page.locator("#note-text mark button")
        popup = self.page.locator("#mention-info")
        outside = self.page.get_by_role("heading", name="Clinical Note Review")
        mentions.first.hover()
        expect(popup).to_be_visible()
        outside.hover()
        expect(popup).to_be_hidden(timeout=1500)

        mentions.first.hover()
        mentions.nth(1).hover()
        expect(popup.locator("#info-code")).to_contain_text(
            self.page.locator("#review-rows tr").nth(1).locator(".code-tag").inner_text()
        )
        popup.hover()
        outside.hover()
        expect(popup).to_be_hidden(timeout=1500)

        mentions.first.hover()
        popup.get_by_role("button", name="Approve (A)").click()
        expect(popup).to_be_hidden()
        expect(self.page.locator("#approved-count")).to_have_text("1")

        mentions.first.focus()
        expect(popup).to_be_visible()
        self.page.keyboard.press("Escape")
        expect(popup).to_be_hidden()
        mentions.nth(1).focus()
        expect(popup).to_be_visible()
        self.page.keyboard.press("Tab")
        expect(popup).to_be_hidden()

    def test_startup_and_switching_between_validation_notes(self):
        from playwright.sync_api import expect

        ids = sorted(self.dataset.splits["validation"])
        self.assertEqual(self.page.title(), "Clinical Note Review")
        expect(self.page.locator("#validation-options option")).to_have_count(len(ids))
        note = self.page.locator("#note-text")
        expect(note).to_have_text(self.dataset.notes[ids[0]])
        self.page.get_by_label("Validation note or custom text").select_option("note:" + ids[1])
        expect(note).to_have_text(self.dataset.notes[ids[1]])

    def test_cached_layout_without_progress_meter_loads_predictions(self):
        from playwright.sync_api import expect

        def cached_layout(route):
            response = route.fetch()
            self.assertIn("no-cache", response.headers.get("cache-control", ""))
            html = re.sub(r"<progress\b[^>]*>.*?</progress>", "", response.text())
            route.fulfill(response=response, body=html)

        self.page.route(self.url + "/", cached_layout)
        self.page.reload()
        expect(self.page.locator("#completion-meter")).to_have_count(0)
        expect(self.page.locator("#review-rows tr").first).to_be_visible()
        expect(self.page.locator("#note-text mark").first).to_be_visible()

    def test_candidates_filter_and_csv_at_laptop_widths(self):
        from playwright.sync_api import expect

        text = "Heart failure. Renal failure."
        actual = self.page.request.post(self.url + "/api/predict", data={"text": text})
        self.assertEqual(actual.status, 200)
        hit = actual.json()["mentions"][0]
        self.assertEqual(len(hit["candidates"]), 3)
        self.assertEqual(hit["candidates"][0], {
            "concept_id": hit["concept_id"], "preferred_term": hit["concept_name"],
            "confidence": hit["confidence"],
        })

        # Use known confidence values to exercise the exact filter threshold.
        mentions = [
            {
                "start": 0, "end": 13, "text": "Heart failure",
                "concept_id": "84114007", "concept_name": "Heart failure", "confidence": .80,
                "candidates": [
                    {"concept_id": "84114007", "preferred_term": "Heart failure", "confidence": .80},
                    {"concept_id": "42343007", "preferred_term": "Congestive heart failure", "confidence": .65},
                    {"concept_id": "48447003", "preferred_term": "Chronic heart failure", "confidence": .55},
                ],
            },
            {
                "start": 15, "end": 28, "text": "Renal failure",
                "concept_id": "42399005", "concept_name": "Renal failure", "confidence": .79,
                "candidates": [
                    {"concept_id": "42399005", "preferred_term": "Renal failure", "confidence": .79},
                ],
            },
        ]
        self.page.route("**/api/predict", lambda route: route.fulfill(
            json={"note_id": None, "text": text, "mentions": mentions},
        ))
        self.analyze(text)
        rows = self.page.locator("#review-rows tr")
        expect(rows).to_have_count(2)
        first, second = rows.nth(0), rows.nth(1)
        first.get_by_role("button", name="Approve").click()
        second.get_by_role("button", name="Reject").click()
        filter_control = self.page.get_by_role("checkbox", name="Show < 0.80 confidence only")
        filter_control.check()
        expect(first).to_be_hidden()
        expect(second).to_be_visible()
        expect(self.page.locator("#note-text mark").first).to_have_class("approved is-filtered")
        expect(self.page.locator("#approved-count")).to_have_text("1")
        filter_control.uncheck()

        candidates = first.get_by_role("combobox")
        expect(candidates.locator("option")).to_have_count(3)
        candidates.select_option("42343007")
        expect(first.locator("td").nth(1)).to_have_text("42343007")
        expect(first.locator("td").nth(3)).to_have_text("65%")
        expect(first.locator("td").nth(3)).to_have_class("confidence-low")
        filter_control.check()
        expect(first).to_be_visible()
        expect(first.get_by_role("button", name="Approve")).to_have_attribute("aria-pressed", "true")
        expect(second.get_by_role("button", name="Reject")).to_have_attribute("aria-pressed", "true")
        candidates.select_option("84114007")
        expect(first).to_be_hidden()
        self.page.locator("#export-toggle").click()
        with self.page.expect_download() as event:
            self.page.get_by_role("button", name="Download CSV").click()
        exported = list(csv.DictReader(io.StringIO(event.value.path().read_text(encoding="utf-8-sig"))))
        self.assertEqual([row["decision"] for row in exported], ["approved", "rejected"])
        filter_control.uncheck()
        candidates.select_option("42343007")
        self.page.locator("#export-toggle").click()
        with self.page.expect_download() as event:
            self.page.get_by_role("button", name="Download CSV").click()
        exported = list(csv.DictReader(io.StringIO(event.value.path().read_text(encoding="utf-8-sig"))))
        self.assertEqual(
            [exported[0][key] for key in ("concept_id", "concept_name", "confidence", "decision")],
            ["42343007", "Congestive heart failure", "0.65", "approved"],
        )

        for _ in range(2):
            for width in (1024, 1200):
                self.page.set_viewport_size({"width": width, "height": 768})
                self.assertTrue(self.page.evaluate("""() => {
                    const selectors = ['.toolbar', '.note-text', '.table-wrap'];
                    return document.documentElement.scrollWidth === innerWidth &&
                        selectors.every(selector => {
                            const box = document.querySelector(selector);
                            return box.scrollWidth <= box.clientWidth;
                        });
                }"""))
                expect(self.page.get_by_role("button", name="Metrics")).to_be_in_viewport(ratio=1)
                expect(first.get_by_role("button", name="Reject")).to_be_in_viewport(ratio=1)
            self.page.locator("#theme-toggle").click()

    def test_metrics_modal_reads_saved_validation_scores(self):
        from playwright.sync_api import expect

        saved = json.loads(BENCHMARK_METRICS_PATH.read_text(encoding="utf-8"))
        trigger = self.page.get_by_role("button", name="Metrics")
        trigger.click()
        dialog = self.page.get_by_role("dialog", name="Validation Metrics")
        expect(dialog).to_be_visible()
        for name in ("Exact String Lookup Baseline", "Entity Linking System"):
            expect(dialog.get_by_role("columnheader", name=name)).to_be_visible()
        for label, key in (
            ("Mention F1", "mention_f1"),
            ("Linking Accuracy", "linking_accuracy"),
            ("End-to-End F1", "end_to_end_f1"),
        ):
            row = dialog.get_by_role("row").filter(has_text=label)
            expect(row.get_by_role("cell")).to_have_text([
                f"{saved['baseline'][key]:.6f}", f"{saved['system'][key]:.6f}",
            ])
        dialog.get_by_role("button", name="Close").click()
        expect(dialog).to_be_hidden()
        expect(trigger).to_be_focused()

    def test_prediction_highlighting_and_review_decisions_survive_note_switch(self):
        from playwright.sync_api import expect

        text = "🩺 <i>Heart failure</i>."
        self.analyze(text)
        note = self.page.locator("#note-text")
        mark = note.locator("mark")
        row = self.page.locator("#review-rows tr")
        expect(note).to_have_text(text)
        expect(mark).to_have_text("Heart failure")
        expect(note.locator("i")).to_have_count(0)
        expect(row).to_contain_text("84114007")

        approve = row.get_by_role("button", name="Approve")
        reject = row.get_by_role("button", name="Reject")
        approve.click()
        expect(approve).to_have_attribute("aria-pressed", "true")
        expect(mark).to_have_class("approved")
        reject.click()
        expect(approve).to_have_attribute("aria-pressed", "false")
        expect(reject).to_have_attribute("aria-pressed", "true")
        expect(mark).to_have_class("rejected")

        first_note = sorted(self.dataset.splits["validation"])[0]
        selector = self.page.get_by_label("Validation note or custom text")
        selector.select_option("note:" + first_note)
        expect(note).to_have_text(self.dataset.notes[first_note])
        self.page.get_by_role("button", name="Paste Custom Note").click()
        expect(reject).to_have_attribute("aria-pressed", "true")
        reject.click()
        expect(reject).to_have_attribute("aria-pressed", "false")
        self.page.locator('#export-toggle').click()
        expect(self.page.get_by_role("button", name="Download CSV")).to_be_disabled()

    def test_reviewed_rows_download_as_csv_and_json(self):
        from playwright.sync_api import expect

        text = "🩺 heart failure; heart failure; heart failure."
        self.analyze(text)
        rows = self.page.locator("#review-rows tr")
        expect(rows).to_have_count(3)
        rows.nth(0).get_by_role("button", name="Approve").click()
        rows.nth(1).get_by_role("button", name="Reject").click()
        for name in ("approved", "rejected", "pending"):
            expect(self.page.locator("#" + name + "-count")).to_have_text("1")
        expect(self.page.locator("#total-count")).to_have_text("3")

        payloads = {}
        for extension in ("CSV", "JSON"):
            self.page.locator("#export-toggle").click()
            with self.page.expect_download() as event:
                self.page.get_by_role("button", name="Download " + extension).click()
            download = event.value
            self.assertEqual(download.suggested_filename, "reviewed-mentions." + extension.lower())
            payloads[extension] = download.path().read_text(encoding="utf-8-sig")

        json_rows = json.loads(payloads["JSON"])
        csv_rows = list(csv.DictReader(io.StringIO(payloads["CSV"])))
        self.assertEqual([row["decision"] for row in json_rows], ["approved", "rejected"])
        self.assertEqual(len(csv_rows), 2)
        for csv_row, json_row in zip(csv_rows, json_rows, strict=True):
            self.assertEqual(csv_row["concept_id"], json_row["concept_id"])
            self.assertEqual(csv_row["decision"], json_row["decision"])
            self.assertEqual(csv_row["start"], str(json_row["start"]))
            self.assertEqual(csv_row["end"], str(json_row["end"]))
            self.assertEqual(text[json_row["start"]:json_row["end"]], json_row["text"])
            self.assertEqual(float(csv_row["confidence"]), json_row["confidence"])
        self.assertEqual(json_rows[0]["start"], 2)


if __name__ == "__main__":
    unittest.main()

