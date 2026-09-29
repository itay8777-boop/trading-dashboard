/* קיצורי מקלדת לבודק: Enter בשדה פרמטר (או Ctrl/Cmd+Enter בכל מקום) = הרצת הבדיקה.
   Dash מעדכן את ערך השדה (debounce) ב-blur/Enter, לכן קודם מוציאים פוקוס ורק אז לוחצים. */
(function () {
    function runNow() {
        var btn = document.getElementById("tester-run-btn");
        if (!btn || btn.disabled) return;
        var tab = document.querySelector(".ptab-sel");
        if (tab && tab.textContent.trim() !== "בודק אסטרטגיות") {
            var t = [...document.querySelectorAll(".ptab")].find(function (e) { return e.textContent.trim() === "בודק אסטרטגיות"; });
            if (t) t.click();
        }
        setTimeout(function () { btn.click(); }, 180);
    }
    document.addEventListener("keydown", function (e) {
        if (e.key !== "Enter") return;
        var el = e.target;
        var inParams = el && el.closest && el.closest(".tester-bar");
        var combo = e.ctrlKey || e.metaKey;
        if (!inParams && !combo) return;
        if (el && el.tagName === "TEXTAREA" && !combo) return;
        e.preventDefault();
        if (el && el.blur) el.blur();
        runNow();
    }, true);
})();
