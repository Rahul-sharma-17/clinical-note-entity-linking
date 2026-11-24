/* ══════════════════════════════════════════════════════════════════════════════
   Theme Manager — Instant FOUC-free theme application + smooth toggle
   ══════════════════════════════════════════════════════════════════════════════ */

let theme = "dark";
try {
  const stored = localStorage.getItem("clinical-review-theme");
  if (stored === "light") theme = "light";
} catch {
  // Storage unavailable — keep default
}

const previewTheme = new URLSearchParams(location.search).get("theme");
if (previewTheme === "light" || previewTheme === "dark") theme = previewTheme;

// Apply immediately (before DOMContentLoaded) to prevent FOUC
document.documentElement.dataset.theme = theme;

document.addEventListener("DOMContentLoaded", () => {
  const toggle = document.querySelector("#theme-toggle");

  function applyTheme() {
    const dark = theme === "dark";

    document.documentElement.dataset.theme = theme;
    toggle.querySelector(".theme-description").textContent =
      dark ? "Theme: Dark" : "Theme: Light";
    toggle.setAttribute("aria-pressed", String(dark));
    toggle.setAttribute("aria-label", dark ? "Switch to light mode" : "Switch to dark mode");
  }

  applyTheme();

  toggle.addEventListener("click", () => {
    theme = theme === "dark" ? "light" : "dark";

    if (previewTheme) {
      const url = new URL(location.href);
      url.searchParams.delete("theme");
      history.replaceState(null, "", url);
    }

    applyTheme();

    try {
      localStorage.setItem("clinical-review-theme", theme);
    } catch {
      // Keep theme for current session only
    }
  });
});
