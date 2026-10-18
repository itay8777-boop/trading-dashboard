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

    /* נקודה כתומה על שדה ששונה ועדיין לא הורץ; הרצה אוטומטית כשהמתג דלוק (לא על 1m) */
    function autoRunOn() {
        var box = document.querySelector("#auto-run input[type=checkbox]");
        var tf = document.querySelector(".tf-btn.active");
        return !!(box && box.checked) && !(tf && tf.textContent.trim() === "1m");
    }
    document.addEventListener("input", function (e) {
        var f = e.target && e.target.closest && e.target.closest(".param-field");
        if (f) f.classList.add("dirty");
    }, true);
    document.addEventListener("change", function (e) {
        var f = e.target && e.target.closest && e.target.closest(".param-field");
        if (!f) return;
        if (autoRunOn()) setTimeout(runNow, 250);
    }, true);
    var mo = new MutationObserver(function () {
        document.querySelectorAll(".param-field.dirty").forEach(function (el) { el.classList.remove("dirty"); });
    });
    var arm = setInterval(function () {
        var lbl = document.getElementById("run-label");
        if (lbl) { mo.observe(lbl, {childList: true, characterData: true, subtree: true}); clearInterval(arm); }
    }, 500);
})();
