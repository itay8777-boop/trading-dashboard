/* גובה הפאנל התחתון: גרירה של המפריד + מזעור/שחזור.
   הכל בצד הדפדפן — שינוי גובה הוא עניין ויזואלי טהור ואין סיבה שיעבור דרך השרת. */
(function () {
    var MIN_PANEL = 39;        // רק רצועת הלשוניות נשארת
    var savedHeight = "";      // הגובה שנבחר ידנית, לשחזור אחרי מזעור

    function init() {
        var panel = document.querySelector(".bottom-panel");
        var ws = document.querySelector(".workspace");
        if (!panel || !ws || panel.querySelector(".panel-resizer")) return false;

        // מפריד לגרירה בראש הפאנל
        var grip = document.createElement("div");
        grip.className = "panel-resizer";
        grip.title = "גרור כדי לשנות גובה · לחיצה כפולה למזעור/שחזור";
        panel.insertBefore(grip, panel.firstChild);

        var dragging = false;
        grip.addEventListener("pointerdown", function (e) {
            dragging = true;
            // לא כל סביבה מאפשרת לתפוס את המצביע; הגרירה עובדת גם בלי זה.
            try { grip.setPointerCapture(e.pointerId); } catch (err) {}
            document.body.style.userSelect = "none";
            e.preventDefault();
        });
        window.addEventListener("pointermove", function (e) {
            if (!dragging) return;
            // הגובה נמדד מתחתית החלון, כך שהגרירה עוקבת אחרי הסמן בדיוק.
            var h = window.innerHeight - e.clientY;
            var max = ws.getBoundingClientRect().height - 120;  // תמיד משאירים גרף נראה
            h = Math.max(MIN_PANEL, Math.min(h, max));
            panel.style.height = h + "px";
            ws.classList.toggle("collapsed", h <= MIN_PANEL + 2);
        });
        window.addEventListener("pointerup", function () {
            if (!dragging) return;
            dragging = false;
            document.body.style.userSelect = "";
            if (!ws.classList.contains("collapsed")) savedHeight = panel.style.height;
        });

        grip.addEventListener("dblclick", function () { toggle(panel, ws); });

        // הכפתור בסרגל מטופל גם ב-Dash (מחליף class). כאן רק מיישרים את הגובה
        // הידני: אחרי מזעור צריך לוותר עליו כדי ש-CSS יקבע, ובשחזור להחזירו.
        new MutationObserver(function () {
            if (ws.classList.contains("collapsed")) {
                if (panel.style.height && panel.style.height !== MIN_PANEL + "px") {
                    savedHeight = panel.style.height;
                }
                panel.style.height = "";
            } else if (savedHeight) {
                panel.style.height = savedHeight;
            }
        }).observe(ws, {attributes: true, attributeFilter: ["class"]});

        return true;
    }

    function toggle(panel, ws) {
        var btn = document.getElementById("panel-toggle");
        if (btn) btn.click();  // עובר דרך Dash כדי שהמצב יישאר מסונכרן
    }

    // ה-layout נבנה על ידי Dash אחרי טעינת הסקריפט, ולכן ממתינים שהאלמנטים יופיעו.
    var tries = 0;
    var timer = setInterval(function () {
        if (init() || ++tries > 60) clearInterval(timer);
    }, 250);
})();
