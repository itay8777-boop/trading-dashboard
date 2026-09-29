/* פריסה זו-לצד-זו: הגבול בין פאנל האסטרטגיה (ימין) לגרף (שמאל) נגרר, נשמר,
   ולחיצה כפולה מכווצת את הפאנל. הכל בדפדפן. */
(function () {
    var KEY = "sl_lab_w";
    function init() {
        var ws = document.querySelector(".workspace");
        var grip = document.getElementById("col-resizer");
        var panel = document.querySelector(".bottom-panel");
        if (!ws || !grip || !panel || grip.dataset.ready) return false;
        grip.dataset.ready = "1";
        var saved = null;
        try { saved = localStorage.getItem(KEY); } catch (e) {}
        if (saved) ws.style.setProperty("--lab-w", saved + "px");
        var dragging = false;
        grip.addEventListener("pointerdown", function (e) {
            dragging = true;
            try { grip.setPointerCapture(e.pointerId); } catch (err) {}
            document.body.style.userSelect = "none";
            e.preventDefault();
        });
        window.addEventListener("pointermove", function (e) {
            if (!dragging) return;
            var rect = ws.getBoundingClientRect();
            var w = rect.right - e.clientX;               // הפאנל בצד ימין
            w = Math.max(360, Math.min(w, rect.width - 320));
            ws.style.setProperty("--lab-w", w + "px");
            ws.classList.remove("collapsed");
        });
        window.addEventListener("pointerup", function () {
            if (!dragging) return;
            dragging = false;
            document.body.style.userSelect = "";
            var v = ws.style.getPropertyValue("--lab-w");
            try { if (v) localStorage.setItem(KEY, parseInt(v, 10)); } catch (e) {}
        });
        grip.addEventListener("dblclick", function () {
            var btn = document.getElementById("panel-toggle");
            if (btn) btn.click();
        });
        return true;
    }
    var timer = setInterval(function () { if (init()) clearInterval(timer); }, 300);
    // G = כיווץ/שחזור הפאנל (מסך גרף מלא); 1/2/3 = שלבים
    document.addEventListener("keydown", function (e) {
        var t = e.target;
        if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
        if (e.key === "g" || e.key === "G") { var b = document.getElementById("panel-toggle"); if (b) b.click(); }
        var steps = {"1": "בדיקה", "2": "עמידות", "3": "נייר"};
        if (steps[e.key]) {
            var tab = [...document.querySelectorAll(".panel-tabs .ptab")].find(function (el) { return el.textContent.trim() === steps[e.key]; });
            if (tab) tab.click();
        }
    });
})();
