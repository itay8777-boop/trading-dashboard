/* ספריית אסטרטגיות (Settings ← Strategy) - כפתור "Strategy library" שפותח
   תפריט בסגנון ה-"My Scripts" של TradingView: רשימת אסטרטגיות שמורות (קליק =
   טעינה + הרצת בקטסט אוטומטית), ו-Save current/Save as new/Delete. הכל בצד
   הלקוח בכוונה - השמירה עצמה עוברת דרך /api/strategies* (app.py), אבל אין שום
   dcc.Store/callback של Dash בתמונה: code-input והגדרות ה-Settings מתעדכנים
   ישירות דרך window.dash_clientside.set_props, בדיוק כמו run-window-store
   שכבר מופעל כך מ-run-btn (ראו app.py). זה נמנע מלהוסיף Output-ים חדשים
   לרכיבים קיימים, שהיה מסתכן בהתנגשות עם callbacks אחרים שכבר עובדים עליהם.

   הכפתור והתפריט הם תוכן רגיל בתוך תת-לשונית "Strategy" (ראו app.py:
   strategy-lib-wrap, position:relative) - לא עוד אלמנט צף שממוקם ב-JS מעל
   כותרת-לשונית: זה היה נחוץ כשה-Pine Editor היה לשונית עצמאית (dcc.Tab לא
   מאפשר תוכן מותאם-אישית בכותרת שלו), אבל אחרי שהוא הפך לתוכן רגיל בתוך
   Settings אין יותר סיבה לטריק הזה - התפריט פשוט position:absolute ביחס
   ל-wrap שלו (ראו style.css), בלי שום מדידת getBoundingClientRect/interval/
   MutationObserver.
   */
(function () {
    var activeStrategyName = null;
    var strategiesCache = [];

    function closeMenu() {
        var menu = document.getElementById("strategy-lib-menu");
        if (menu) menu.style.display = "none";
    }

    function isMenuOpen() {
        var menu = document.getElementById("strategy-lib-menu");
        return !!menu && menu.style.display === "block";
    }

    function escapeHtml(s) {
        return String(s).replace(/[&<>"]/g, function (c) {
            return {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c];
        });
    }

    function escapeAttr(s) {
        return escapeHtml(s).replace(/'/g, "&#39;");
    }

    function renderMenu() {
        var menu = document.getElementById("strategy-lib-menu");
        if (!menu) return;
        var html = "";
        if (!strategiesCache.length) {
            html += '<div class="strategy-lib-empty">No saved strategies</div>';
        } else {
            strategiesCache.forEach(function (s) {
                var active = s.name === activeStrategyName;
                html += '<div class="strategy-lib-item' + (active ? " strategy-lib-active" : "") +
                    '" data-strategy-name="' + escapeAttr(s.name) + '">' + escapeHtml(s.name) + "</div>";
            });
        }
        html += '<div class="strategy-lib-divider"></div>';
        html += '<div class="strategy-lib-action" data-action="save-current">Save current strategy</div>';
        html += '<div class="strategy-lib-action" data-action="save-new">Save as new strategy</div>';
        var deleteDisabled = !activeStrategyName;
        html += '<div class="strategy-lib-action' + (deleteDisabled ? " disabled" : "") +
            '" data-action="delete">Delete strategy</div>';
        menu.innerHTML = html;
    }

    async function fetchStrategies() {
        try {
            var res = await fetch("/api/strategies");
            var data = await res.json();
            strategiesCache = data.strategies || [];
        } catch (err) {
            strategiesCache = [];
        }
    }

    async function openMenu() {
        await fetchStrategies();
        renderMenu();
        var menu = document.getElementById("strategy-lib-menu");
        if (menu) menu.style.display = "block";
    }

    function toggleMenu() {
        if (isMenuOpen()) { closeMenu(); return; }
        openMenu();
    }

    // ה-wrap הוא position:relative (ראו style.css: strategy-lib-wrap) - גם
    // התפריט וגם ה-toast ממוקמים ביחס אליו ישירות ב-CSS (top/left), בלי שום
    // מדידת getBoundingClientRect כאן.
    function showToast(text) {
        var wrap = document.querySelector(".strategy-lib-wrap");
        if (!wrap) return;
        var toast = document.getElementById("strategy-lib-toast");
        if (!toast) {
            toast = document.createElement("div");
            toast.id = "strategy-lib-toast";
            toast.className = "strategy-lib-toast";
            wrap.appendChild(toast);
        }
        toast.textContent = text;
        toast.classList.add("show");
        clearTimeout(toast._hideTimer);
        toast._hideTimer = setTimeout(function () { toast.classList.remove("show"); }, 1800);
    }

    function currentSettings() {
        function num(id, dflt) {
            var el = document.getElementById(id);
            var v = el ? parseFloat(el.value) : NaN;
            return isNaN(v) ? dflt : v;
        }
        return {
            capital: num("account-value-input", 100000),
            contracts: num("contracts-input", 1),
            multiplier: num("multiplier-input", 2),
            slippage: num("slippage-input", 2),
            commission: num("settings-commission-input", 0),
        };
    }

    function openSaveModal(prefillName) {
        var overlay = document.getElementById("strategy-save-modal-overlay");
        var input = document.getElementById("strategy-save-name-input");
        if (!overlay || !input) return;
        input.value = prefillName || "";
        overlay.classList.add("open");
        setTimeout(function () { input.focus(); input.select(); }, 30);
    }

    function closeSaveModal() {
        var overlay = document.getElementById("strategy-save-modal-overlay");
        if (overlay) overlay.classList.remove("open");
    }

    async function confirmSave() {
        var input = document.getElementById("strategy-save-name-input");
        var name = input ? input.value.trim() : "";
        if (!name) return;
        var code = document.getElementById("code-input");
        var payload = {name: name, code: code ? code.value : "", settings: currentSettings()};
        try {
            var res = await fetch("/api/strategies/save", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify(payload),
            });
            var data = await res.json();
            if (data.ok) {
                activeStrategyName = name;
                showToast("Saved ✓");
            }
        } catch (err) { /* חיבור נכשל - פשוט לא שומרים, בלי לקרוס את הממשק */ }
        closeSaveModal();
    }

    async function deleteActive() {
        if (!activeStrategyName) return;
        if (!window.confirm('Delete "' + activeStrategyName + '"?')) return;
        try {
            await fetch("/api/strategies/delete", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({name: activeStrategyName}),
            });
            showToast("Deleted");
        } catch (err) { /* ראו confirmSave */ }
        activeStrategyName = null;
    }

    function loadAndRun(strategy) {
        var setProps = window.dash_clientside && window.dash_clientside.set_props;
        if (!setProps) return;
        activeStrategyName = strategy.name;
        closeMenu();

        setProps("code-input", {value: strategy.code || ""});
        var s = strategy.settings || {};
        if (s.capital != null) setProps("account-value-input", {value: s.capital});
        if (s.contracts != null) setProps("contracts-input", {value: s.contracts});
        if (s.multiplier != null) setProps("multiplier-input", {value: s.multiplier});
        if (s.slippage != null) setProps("slippage-input", {value: s.slippage});
        if (s.commission != null) setProps("settings-commission-input", {value: s.commission});

        // עוברים ללשונית Tester *לפני* הפעלת הבקטסט - כדי שספינר הטעינה (dcc.Loading
        // סביב key-stats/result-body) יופיע בכלל; run_or_switch ב-app.py מחליף
        // ללשונית הזו רק *אחרי* שהוא מסיים, בדיוק כמו שכפתור "Add to chart" הרגיל
        // עושה זאת מראש דרך ה-clientside_callback הנפרד שלו.
        setProps("panel-tabs", {value: "tester"});
        // עיכוב קטן לפני run-window-store: setProps("code-input", ...) למעלה
        // מפעיל round-trip לשרת (update_strategy_inputs ב-app.py, שבונה מחדש
        // את strategy-inputs-container לברירות-המחדל של האסטרטגיה *החדשה*) -
        // בלי העיכוב, run-window-store עלול "לדרוס" ולהגיע לפני שה-round-trip
        // הזה הסתיים, ואז run_or_switch קורא State עדיין עם שדות-הקלט של
        // האסטרטגיה ה*קודמת* (למשל STOP_LOSS_PTS שנשאר על הערך הישן, כי שם
        // השדה זהה בין שתי האסטרטגיות אבל ברירת-המחדל שונה) - נמצא בפועל
        // בדיוק ככה. app.py:run_or_switch מכיל גם הגנה משלימה בצד-שרת (בודקת
        // שקבוצת שמות-השדות תואמת את מה ש-parse_strategy_inputs מפרש מהקוד
        // הנוכחי, ומתעלמת מ-overrides אם לא) - העיכוב כאן מקטין את הסיכוי
        // שבכלל יגיע למצב הזה, לא מחליף את ההגנה בצד-שרת.
        setTimeout(function () {
            // run-window-store הוא ה-Input שמפעיל את run_or_switch (ראו app.py);
            // הבקטסט עצמו תמיד רץ על כל ההיסטוריה בלי קשר לתוכן כאן - רק
            // ה-"n" הייחודי חשוב, כדי ש-Dash יזהה את זה כשינוי נתונים אמיתי
            // בכל קליק (בדיוק כמו run-btn).
            setProps("run-window-store", {data: {from: null, to: null, n: Date.now()}});
        }, 150);
    }

    document.addEventListener("click", function (e) {
        if (e.target.closest("#strategy-lib-arrow")) { toggleMenu(); return; }

        var item = e.target.closest("#strategy-lib-menu .strategy-lib-item");
        if (item) {
            var name = item.getAttribute("data-strategy-name");
            var strategy = strategiesCache.filter(function (s) { return s.name === name; })[0];
            if (strategy) loadAndRun(strategy);
            return;
        }

        var action = e.target.closest("#strategy-lib-menu .strategy-lib-action");
        if (action) {
            if (action.classList.contains("disabled")) return;
            var act = action.getAttribute("data-action");
            closeMenu();
            if (act === "save-current") openSaveModal(activeStrategyName);
            else if (act === "save-new") openSaveModal("");
            else if (act === "delete") deleteActive();
            return;
        }

        if (e.target.closest("#strategy-save-close") || e.target.closest("#strategy-save-cancel")) {
            closeSaveModal();
            return;
        }
        if (e.target.id === "strategy-save-modal-overlay") { closeSaveModal(); return; }
        if (e.target.closest("#strategy-save-confirm")) { confirmSave(); return; }

        if (!e.target.closest("#strategy-lib-menu") && !e.target.closest("#strategy-lib-arrow")) {
            closeMenu();
        }
    });

    document.addEventListener("keydown", function (e) {
        if (e.key === "Escape") { closeMenu(); closeSaveModal(); }
        if (e.key === "Enter" && e.target && e.target.id === "strategy-save-name-input") {
            confirmSave();
        }
    });

})();
